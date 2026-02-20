"""
ReAct Agent for Research Knowledge Graph
=========================================
Pattern: Reason → Act → Observe → Repeat → Answer

The agent has 6 tools:
  1. keyword_search     — search papers by keyword in title/abstract
  2. author_search      — find papers by a specific author
  3. institution_search — papers from a university/institute
  4. topic_search       — papers about a specific topic
  5. cypher_search      — LLM generates custom Cypher for complex filters
  6. get_statistics     — overall graph stats (counts, averages)

Flow:
  User question
      ↓
  Agent asks LLM: "Which tool? What args? Why?"
      ↓
  Execute tool → get observation
      ↓
  Agent asks LLM: "Is this enough to answer? Or need another tool?"
      ↓  (max 3 iterations)
  Generate final natural language answer
      ↓
  Return answer + full thought chain (for Streamlit UI)
"""

from __future__ import annotations   # Python 3.9 compatibility

import json
import re
import time
import requests
from typing import Any, Optional
from loguru import logger

from config import Config, Neo4jConnection
from search_kg import KnowledgeGraphSearch
from cypher_generator import CypherGenerator, get_schema_description


# ─────────────────────────────────────────────
# TOOL DEFINITIONS  (shown to LLM)
# ─────────────────────────────────────────────

TOOLS_DESCRIPTION = """
You have access to these tools to search a research paper knowledge graph:

1. keyword_search(query: str)
   Use when: General question about a topic, concept, or theme
   Example: keyword_search("graph neural networks")

2. author_search(name: str)
   Use when: Question mentions a specific researcher or asks "who wrote"
   Example: author_search("Sarah Johnson")

3. institution_search(name: str)
   Use when: Question mentions a university, college, or research lab
   Example: institution_search("IIT Tirupati")

4. topic_search(topic: str)
   Use when: Question asks about a specific research field or topic node
   Example: topic_search("Deep Learning")

5. cypher_search(question: str)
   Use when: Question has numeric filters (citations > X), date filters (after 2020),
             comparisons (more than, less than), or complex multi-condition queries
   Example: cypher_search("papers with more than 500 citations published after 2021")

6. get_statistics()
   Use when: Question asks "how many", "total", "count", "statistics", "overview"
   Example: get_statistics()

7. FINISH(answer: str)
   Use when: You have enough information to answer the question completely
   Example: FINISH("Based on the results, the top papers are...")
"""


# ─────────────────────────────────────────────
# REACT AGENT
# ─────────────────────────────────────────────

class ResearchAgent:
    """
    ReAct Agent — Reason + Act loop for research paper Q&A.

    Each "step" consists of:
      - Thought : LLM reasons about what to do next
      - Action  : LLM picks a tool + args
      - Observation: Tool executes and returns results

    The loop stops when:
      - LLM picks FINISH action
      - Max steps reached (safety limit)
      - Tool returns sufficient results
    """

    OLLAMA_URL = "http://localhost:11434/api/generate"
    MODEL      = "llama3.2"
    MAX_STEPS  = 3          # Max tool calls before forcing an answer

    def __init__(self):
        self.searcher          = KnowledgeGraphSearch()
        self.searcher.connect()
        self.cypher_generator  = CypherGenerator("ollama")
        self.config            = Config()

        # Tool registry — maps tool name → method
        self.tools = {
            "keyword_search"     : self._tool_keyword_search,
            "author_search"      : self._tool_author_search,
            "institution_search" : self._tool_institution_search,
            "topic_search"       : self._tool_topic_search,
            "cypher_search"      : self._tool_cypher_search,
            "get_statistics"     : self._tool_get_statistics,
        }

        logger.info(f"✓ ResearchAgent initialized | model={self.MODEL} | tools={list(self.tools.keys())}")

    def close(self):
        self.searcher.close()

    # ─────────────────────────────────────────
    # MAIN ENTRY: run()
    # ─────────────────────────────────────────

    def run(self, question: str) -> dict:
        """
        Run the ReAct loop for a user question.

        Returns:
            {
              "question"    : original question,
              "answer"      : final natural language answer,
              "thought_chain": list of {thought, action, args, observation},
              "steps_taken" : number of tool calls made,
              "latency_s"   : total time in seconds,
              "success"     : True/False
            }
        """
        logger.info(f"\n{'='*60}\nAgent Question: {question}\n{'='*60}")
        start_time   = time.time()
        thought_chain = []
        all_observations = []

        for step in range(self.MAX_STEPS):
            logger.info(f"\n--- Step {step + 1}/{self.MAX_STEPS} ---")

            # ── Ask LLM: what should I do next? ──
            action_decision = self._decide_action(
                question       = question,
                thought_chain  = thought_chain,
                step_number    = step + 1,
            )

            tool_name = action_decision.get("tool", "keyword_search")
            tool_args = action_decision.get("args", {})
            thought   = action_decision.get("thought", "")

            logger.info(f"Thought : {thought}")
            logger.info(f"Action  : {tool_name}({tool_args})")

            # ── FINISH action — generate final answer ──
            if tool_name == "FINISH":
                logger.info("Agent decided it has enough information → generating answer")
                answer = self._generate_final_answer(question, all_observations)
                elapsed = round(time.time() - start_time, 2)
                return {
                    "question"     : question,
                    "answer"       : answer,
                    "thought_chain": thought_chain,
                    "steps_taken"  : step,
                    "latency_s"    : elapsed,
                    "success"      : True,
                }

            # ── Execute tool ──
            observation = self._execute_tool(tool_name, tool_args)
            obs_summary = self._summarize_observation(observation)

            logger.info(f"Observation: {obs_summary}")

            # Record step
            thought_chain.append({
                "step"       : step + 1,
                "thought"    : thought,
                "action"     : tool_name,
                "args"       : tool_args,
                "observation": obs_summary,
                "raw_results": observation,
            })

            all_observations.append({
                "tool"       : tool_name,
                "args"       : tool_args,
                "results"    : observation,
                "summary"    : obs_summary,
            })

            # ── Early exit: if we got good results, no need for more steps ──
            if self._has_sufficient_results(observation) and step >= 1:
                logger.info("Sufficient results found — stopping early")
                break

        # ── Generate final answer from all collected observations ──
        answer  = self._generate_final_answer(question, all_observations)
        elapsed = round(time.time() - start_time, 2)

        logger.info(f"\n✓ Agent done | steps={len(thought_chain)} | latency={elapsed}s")

        return {
            "question"     : question,
            "answer"       : answer,
            "thought_chain": thought_chain,
            "steps_taken"  : len(thought_chain),
            "latency_s"    : elapsed,
            "success"      : bool(answer and "couldn't find" not in answer.lower()),
        }

    # ─────────────────────────────────────────
    # STEP 1: DECIDE ACTION (LLM reasoning)
    # ─────────────────────────────────────────

    def _decide_action(
        self,
        question: str,
        thought_chain: list,
        step_number: int,
    ) -> dict:
        """
        Ask the LLM which tool to use next and with what arguments.
        Returns: {"thought": "...", "tool": "...", "args": {...}}
        """

        # ── Rule-based pre-check (faster + more reliable than LLM for obvious cases) ──
        if step_number == 1:
            q = question.lower()

            # "how many / total / count / statistics" → always get_statistics
            if any(w in q for w in ["how many", "total", "count", "statistics", "overview"]):
                return {
                    "thought": "Question asks for counts/statistics — using get_statistics directly.",
                    "tool"   : "get_statistics",
                    "args"   : {},
                }

            # Institution keywords → institution_search
            if any(w in q for w in ["iit", "mit", "stanford", "university", "institution", "college"]):
                import re as _re
                inst_match = _re.search(r"from ([A-Za-z ]+?)(?:\s+(?:paper|author|research)|$)", question, _re.I)
                inst_name  = inst_match.group(1).strip() if inst_match else "IIT"
                return {
                    "thought": "Question asks about an institution — using institution_search.",
                    "tool"   : "institution_search",
                    "args"   : {"name": inst_name},
                }

        # ── If previous step already returned results → FINISH now ──
        if step_number >= 2 and thought_chain:
            last_obs = thought_chain[-1].get("observation", "")
            if "Found" in last_obs and "0 results" not in last_obs:
                return {
                    "thought": "I already have sufficient results from the previous step.",
                    "tool"   : "FINISH",
                    "args"   : {},
                }

        # Build history of previous steps for context
        history = ""
        if thought_chain:
            history = "\n\nPREVIOUS STEPS:\n"
            for step in thought_chain:
                history += f"Step {step['step']}:\n"
                history += f"  Thought: {step['thought']}\n"
                history += f"  Action : {step['action']}({step['args']})\n"
                history += f"  Result : {step['observation']}\n"

        prompt = f"""You are a research assistant agent. Your job is to find information about research papers.

{TOOLS_DESCRIPTION}
{history}

QUESTION: {question}
STEP: {step_number} of {self.MAX_STEPS}

{"IMPORTANT: This is your last step. You MUST use FINISH now." if step_number == self.MAX_STEPS else ""}

Think about what information you need, then choose ONE tool.
If you already have enough information from previous steps, use FINISH.

Respond with ONLY this JSON:
{{
  "thought": "I need to ... because ...",
  "tool": "tool_name_here",
  "args": {{"arg_name": "arg_value"}}
}}

For get_statistics, use: "args": {{}}
For FINISH, use: "args": {{"answer": "brief summary"}}
"""

        raw = self._call_llm(prompt, temperature=0.2, max_tokens=250)
        result = self._parse_json(raw)

        # Fallback if JSON parsing fails
        if not result:
            logger.warning("Action parsing failed — using keyword_search as fallback")
            # Extract key words from question for fallback
            words = [w for w in question.split() if len(w) > 3]
            query = " ".join(words[:4]) if words else question
            result = {
                "thought": "Falling back to keyword search",
                "tool"   : "keyword_search",
                "args"   : {"query": query},
            }

        return result

    # ─────────────────────────────────────────
    # STEP 2: EXECUTE TOOL
    # ─────────────────────────────────────────

    def _execute_tool(self, tool_name: str, tool_args: dict) -> Any:
        """Execute the chosen tool and return raw results"""
        try:
            if tool_name not in self.tools:
                logger.warning(f"Unknown tool '{tool_name}' — using keyword_search")
                tool_name = "keyword_search"
                tool_args = {"query": list(tool_args.values())[0] if tool_args else ""}

            result = self.tools[tool_name](**tool_args)
            return result

        except TypeError as e:
            # Wrong args passed — try with just the first value
            logger.warning(f"Tool arg mismatch for {tool_name}: {e} — retrying")
            try:
                first_val = list(tool_args.values())[0] if tool_args else ""
                return self.tools[tool_name](first_val)
            except Exception as e2:
                logger.error(f"Tool execution failed: {e2}")
                return []

        except Exception as e:
            logger.error(f"Tool '{tool_name}' failed: {e}")
            return []

    # ─────────────────────────────────────────
    # STEP 3: GENERATE FINAL ANSWER
    # ─────────────────────────────────────────

    def _generate_final_answer(self, question: str, observations: list) -> str:
        """Generate a natural language answer from all tool observations"""

        if not observations:
            return "I couldn't find relevant information for your question in the knowledge graph."

        # Build context from all observations
        context_parts = []
        for i, obs in enumerate(observations, 1):
            context_parts.append(f"\n[Tool {i}: {obs['tool']}({obs['args']})]")
            context_parts.append(obs["summary"])

        context = "\n".join(context_parts)

        prompt = f"""You are a research assistant. Answer ONLY using the exact data shown below.

STRICT RULES:
- NEVER invent or calculate numbers not present in the data
- If data shows statistics like "5 papers", report exactly that number
- Do not guess, assume duplicates, or add/subtract from shown numbers
- Be concise and factual, cite exact titles and numbers from data

DATA RETRIEVED:
{context}

QUESTION: {question}

ANSWER (use only numbers and titles shown in DATA above):"""

        answer = self._call_llm(prompt, temperature=0.4, max_tokens=500)
        return answer.strip() if answer else "Could not generate an answer from the retrieved data."

    # ─────────────────────────────────────────
    # TOOL IMPLEMENTATIONS
    # ─────────────────────────────────────────

    def _tool_keyword_search(self, query: str = "") -> list:
        # Guard: if empty query, return all papers
        if not query or not query.strip():
            logger.warning("  keyword_search called with empty query — returning top papers instead")
            return self.searcher.get_most_cited_papers(10)

        logger.info(f"  🔍 keyword_search('{query}')")
        results = self.searcher.search_papers_by_keyword(query)
        logger.info(f"  → {len(results)} papers found")

        # If no results with full query, retry with first meaningful word
        if not results and len(query.split()) > 2:
            short_query = query.split()[0]  # e.g. "attention mechanism in deep learning" → "attention"
            logger.info(f"  Retrying with shorter query: '{short_query}'")
            results = self.searcher.search_papers_by_keyword(short_query)
            logger.info(f"  → {len(results)} papers found with short query")

        return results

    def _tool_author_search(self, name: str) -> list:
        logger.info(f"  👤 author_search('{name}')")
        results = self.searcher.search_by_author(name)
        logger.info(f"  → {len(results)} papers found")
        return results

    def _tool_institution_search(self, name: str) -> list:
        logger.info(f"  🏛  institution_search('{name}')")
        results = self.searcher.search_by_institution(name)
        logger.info(f"  → {len(results)} papers found")
        return results

    def _tool_topic_search(self, topic: str) -> list:
        logger.info(f"  🏷  topic_search('{topic}')")
        results = self.searcher.search_by_topic(topic)
        logger.info(f"  → {len(results)} papers found")
        return results

    def _tool_get_statistics(self) -> dict:
        logger.info("  📊 get_statistics()")
        stats = self.searcher.get_statistics()
        logger.info(f"  → stats: {stats}")
        return stats

    def _tool_cypher_search(self, question: str) -> list:
        """LLM generates custom Cypher for complex queries"""
        logger.info(f"  ⚙️  cypher_search('{question[:60]}...')")
        schema = get_schema_description()
        cypher_result = self.cypher_generator.generate_cypher(question, schema)

        if not cypher_result.get("cypher"):
            logger.warning("  Cypher generation failed — falling back to keyword search")
            return self.searcher.search_papers_by_keyword(question)

        cypher = cypher_result["cypher"]
        logger.info(f"  Generated: {cypher[:80]}...")

        try:
            with Neo4jConnection(self.config) as db:
                results = db.execute_query(cypher)
                logger.info(f"  → {len(results)} results")
                return results
        except Exception as e:
            logger.error(f"  Cypher execution failed: {e}")
            return self.searcher.search_papers_by_keyword(question)

    # ─────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────

    def _summarize_observation(self, observation: Any) -> str:
        """Convert raw tool output into a readable summary string"""

        # Statistics dict
        if isinstance(observation, dict) and "total_papers" in observation:
            return (
                f"Statistics: {observation.get('total_papers', 0)} papers, "
                f"{observation.get('total_authors', 0)} authors, "
                f"{observation.get('total_topics', 0)} topics, "
                f"{observation.get('total_institutions', 0)} institutions, "
                f"avg citations: {round(observation.get('avg_citations') or 0, 1)}"
            )

        # List of papers
        if isinstance(observation, list):
            if not observation:
                return "No results found."

            lines = [f"Found {len(observation)} results:"]
            for i, item in enumerate(observation[:5], 1):   # Show top 5 in summary
                title     = item.get("title", item.get("p.title", "Unknown"))
                citations = item.get("citations", item.get("p.citation_count", "?"))
                year      = item.get("year", item.get("p.year", ""))
                author    = item.get("author", "")
                year_str  = f" ({year})" if year else ""
                auth_str  = f" — {author}" if author else ""
                lines.append(f"  {i}. {title[:70]}{year_str}{auth_str} [{citations} citations]")

            if len(observation) > 5:
                lines.append(f"  ... and {len(observation) - 5} more")

            return "\n".join(lines)

        return str(observation)[:300]

    def _has_sufficient_results(self, observation: Any) -> bool:
        """Check if tool returned enough results to stop the loop early"""
        if isinstance(observation, dict) and "total_papers" in observation:
            return True    # Stats always sufficient
        if isinstance(observation, list) and len(observation) >= 3:
            return True    # 3+ papers is sufficient
        return False

    def _call_llm(
        self,
        prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 400,
    ) -> str:
        """Call Ollama LLM and return response text"""
        try:
            response = requests.post(
                self.OLLAMA_URL,
                json={
                    "model" : self.MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens,
                    },
                },
                timeout=90,
            )
            if response.status_code == 200:
                return response.json()["response"]
            else:
                logger.error(f"Ollama HTTP {response.status_code}")
                return ""
        except requests.exceptions.ConnectionError:
            logger.error("Ollama not running — start with: ollama serve")
            return ""
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return ""

    def _parse_json(self, text: str) -> Optional[dict]:
        """Robustly extract JSON from LLM response"""
        text = text.strip()

        # Try direct parse
        try:
            return json.loads(text)
        except Exception:
            pass

        # Strip markdown fences
        fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
        if fence:
            try:
                return json.loads(fence.group(1).strip())
            except Exception:
                pass

        # Find first { ... } block
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                pass

        return None


# ─────────────────────────────────────────────
# INTERACTIVE MODE
# ─────────────────────────────────────────────

def interactive_agent():
    print("\n" + "=" * 70)
    print("🤖  REACT AGENT — Research Knowledge Graph")
    print("=" * 70)
    print("  The agent REASONS before searching, picks the right tool,")
    print("  and can make multiple tool calls to answer complex questions.")
    print("=" * 70)
    print("\nExample questions:")
    print("  • papers with more than 100 citations")
    print("  • who are the authors from IIT Tirupati?")
    print("  • papers about deep learning published after 2020")
    print("  • how many papers are in the database?")
    print("  • papers by Dr. Sarah Johnson")
    print()

    agent = ResearchAgent()

    try:
        while True:
            question = input("🔍 Your question (or 'quit'): ").strip()

            if question.lower() in ["quit", "exit", "q"]:
                break
            if not question:
                continue

            print("\n⏳ Agent thinking...\n")
            result = agent.run(question)

            # ── Print thought chain ──
            print("─" * 70)
            print("🧠 AGENT THOUGHT CHAIN:")
            print("─" * 70)
            for step in result["thought_chain"]:
                print(f"\n  Step {step['step']}:")
                print(f"  💭 Thought : {step['thought']}")
                print(f"  🔧 Action  : {step['action']}({step['args']})")
                print(f"  👁  Observed: {step['observation'][:200]}")

            # ── Print final answer ──
            print("\n" + "=" * 70)
            print("💡 FINAL ANSWER:")
            print("=" * 70)
            print(result["answer"])
            print()
            print(f"  ⏱  Latency : {result['latency_s']}s")
            print(f"  🔢 Steps   : {result['steps_taken']}")
            print(f"  ✅ Success : {result['success']}")
            print("=" * 70 + "\n")

    finally:
        agent.close()
        print("👋 Goodbye!")


if __name__ == "__main__":
    interactive_agent()