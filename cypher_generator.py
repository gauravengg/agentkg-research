"""
LLM Cypher Query Generator
Generates Neo4j Cypher queries from natural language

FIXES APPLIED (Step 1):
  FIX-1: Changed model from "llama2" → "llama3.2" (llama2 is bad at JSON output)
  FIX-2: Added retry logic — retries once with simpler prompt if JSON parse fails
  FIX-3: Added Cypher safety validator — blocks DROP, DELETE, DETACH to prevent data loss
  FIX-4: Better JSON extraction — handles more edge cases in LLM response formatting
"""

import json
import re
import requests
from loguru import logger


# ─────────────────────────────────────────────
# DANGEROUS CYPHER KEYWORDS TO BLOCK
# (FIX-3: safety guard)
# ─────────────────────────────────────────────
BLOCKED_CYPHER_KEYWORDS = ["DROP", "DELETE", "DETACH", "REMOVE", "SET", "CREATE", "MERGE"]


class CypherGenerator:
    """
    Uses LLM to generate Cypher queries from natural language
    """

    def __init__(self, llm_provider="ollama"):
        self.llm_provider = llm_provider
        # FIX-1: was "llama2", now "llama3.2" — much better at structured JSON output
        self.model = "llama3.2"

    def generate_cypher(self, user_question: str, schema_info: str) -> dict:
        """
        Generate a READ-ONLY Cypher query from a natural language question.

        Returns:
            dict with keys: 'cypher', 'explanation', optionally 'error'
        """
        prompt = self._build_prompt(user_question, schema_info)

        try:
            if self.llm_provider == "ollama":
                raw_response = self._call_ollama(prompt)
            else:
                return {"cypher": None, "explanation": "Only Ollama is supported", "error": "unsupported provider"}

            # FIX-4: Robust JSON extraction
            result = self._extract_json(raw_response)

            if not result or "cypher" not in result:
                # FIX-2: Retry once with a simpler prompt
                logger.warning("First attempt failed, retrying with simpler prompt...")
                simple_prompt = self._build_simple_prompt(user_question)
                raw_response2 = self._call_ollama(simple_prompt)
                result = self._extract_json(raw_response2)

            if not result or "cypher" not in result:
                return {"cypher": None, "explanation": "Failed after retry", "error": "JSON parse failed both attempts"}

            # FIX-3: Safety check — block write operations
            cypher = result["cypher"]
            safety_error = self._safety_check(cypher)
            if safety_error:
                logger.error(f"Blocked unsafe Cypher: {safety_error}")
                return {"cypher": None, "explanation": safety_error, "error": "unsafe_query"}

            logger.info(f"✓ Generated Cypher: {cypher[:100]}...")
            return result

        except Exception as e:
            logger.error(f"Cypher generation failed: {e}")
            return {"cypher": None, "explanation": str(e), "error": str(e)}

    # ─────────────────────────────────────────
    # PROMPTS
    # ─────────────────────────────────────────

    def _build_prompt(self, question: str, schema: str) -> str:
        return f"""You are a Neo4j expert. Generate a READ-ONLY Cypher query.

GRAPH SCHEMA:
{schema}

QUESTION: {question}

Rules:
1. Only use MATCH, WHERE, RETURN, ORDER BY, LIMIT
2. NEVER use CREATE, DELETE, SET, MERGE, REMOVE, DROP
3. Always LIMIT to 20 results max
4. Use toLower() for case-insensitive string matching
5. Return meaningful property names

Respond with ONLY this JSON and nothing else:
{{"cypher": "MATCH ...", "explanation": "one sentence explaining what this does"}}"""

    def _build_simple_prompt(self, question: str) -> str:
        """FIX-2: Simpler fallback prompt for retry"""
        return f"""Write a Neo4j Cypher MATCH query for: "{question}"

Use nodes: Paper (title, year, citation_count), Author (name), Topic (name), Institution (name)
Use relationships: WROTE, ABOUT, CITES, AFFILIATED_WITH

Return JSON only:
{{"cypher": "MATCH (p:Paper) RETURN p.title LIMIT 10", "explanation": "brief explanation"}}"""

    # ─────────────────────────────────────────
    # JSON EXTRACTION (FIX-4)
    # ─────────────────────────────────────────

    def _extract_json(self, text: str):
        """
        Robustly extract JSON from LLM response.
        Handles: raw JSON, ```json blocks, ```  blocks, JSON embedded in prose.
        """
        text = text.strip()

        # Strategy 1: Try direct parse
        try:
            return json.loads(text)
        except Exception:
            pass

        # Strategy 2: Strip markdown code blocks
        # handles ```json ... ``` and ``` ... ```
        code_block = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
        if code_block:
            try:
                return json.loads(code_block.group(1).strip())
            except Exception:
                pass

        # Strategy 3: Find first { ... } JSON object in response
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except Exception:
                pass

        logger.warning(f"Could not extract JSON from response: {text[:200]}")
        return None

    # ─────────────────────────────────────────
    # SAFETY CHECK (FIX-3)
    # ─────────────────────────────────────────

    def _safety_check(self, cypher: str):
        """
        Returns an error message if the Cypher contains write operations,
        None if it is safe to execute.
        """
        cypher_upper = cypher.upper()
        for keyword in BLOCKED_CYPHER_KEYWORDS:
            # Only block if keyword appears as a standalone word, not inside MATCH
            if re.search(rf"\b{keyword}\b", cypher_upper):
                return f"Blocked: Cypher contains write operation '{keyword}'"
        return None

    # ─────────────────────────────────────────
    # OLLAMA CALL
    # ─────────────────────────────────────────

    def _call_ollama(self, prompt: str) -> str:
        """Call local Ollama LLM"""
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": self.model,          # FIX-1: llama3.2 not llama2
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,        # Low temp for precise structured output
                        "num_predict": 300,        # Enough for a Cypher + explanation
                    }
                },
                timeout=45,
            )

            if response.status_code == 200:
                return response.json()["response"]
            else:
                logger.error(f"Ollama HTTP {response.status_code}")
                return ""

        except requests.exceptions.ConnectionError:
            logger.error("Ollama not running. Start with: ollama serve")
            return ""
        except Exception as e:
            logger.error(f"Ollama error: {e}")
            return ""


def get_schema_description() -> str:
    """Return a description of the graph schema for LLM prompts"""
    return """
NODE TYPES:
- Paper:       paper_id, title, abstract, year (int), citation_count (int)
- Author:      author_id, name, h_index (int)
- Institution: inst_id, name, country
- Topic:       name

RELATIONSHIPS:
- (Author)-[:WROTE]->(Paper)
- (Author)-[:AFFILIATED_WITH]->(Institution)
- (Paper)-[:ABOUT]->(Topic)
- (Paper)-[:CITES]->(Paper)

EXAMPLE QUERIES:
  Papers after 2020:
    MATCH (p:Paper) WHERE p.year > 2020 RETURN p.title, p.year ORDER BY p.year DESC LIMIT 20

  Papers from IIT:
    MATCH (i:Institution)<-[:AFFILIATED_WITH]-(a:Author)-[:WROTE]->(p:Paper)
    WHERE toLower(i.name) CONTAINS 'iit'
    RETURN p.title, a.name, i.name LIMIT 20

  Most cited:
    MATCH (p:Paper) RETURN p.title, p.citation_count ORDER BY p.citation_count DESC LIMIT 10

  Papers with >500 citations about deep learning:
    MATCH (p:Paper)-[:ABOUT]->(t:Topic)
    WHERE p.citation_count > 500 AND toLower(t.name) CONTAINS 'deep learning'
    RETURN p.title, p.citation_count LIMIT 20
"""


# ─────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────
if __name__ == "__main__":
    generator = CypherGenerator("ollama")
    schema = get_schema_description()

    test_questions = [
        "Find papers from IIT with more than 500 citations",
        "Who are authors from IIT Tirupati?",
        "Papers about deep learning published after 2020",
    ]

    for q in test_questions:
        print(f"\n{'='*70}\nQuestion: {q}\n{'='*70}")
        result = generator.generate_cypher(q, schema)
        if result["cypher"]:
            print(f"✓ Cypher:      {result['cypher']}")
            print(f"  Explanation: {result.get('explanation', '')}")
        else:
            print(f"✗ Failed: {result.get('error', 'Unknown error')}")