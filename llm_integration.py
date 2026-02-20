"""
Hybrid LLM + Knowledge Graph System

FIXES APPLIED (Step 1):
  FIX-1: Changed model from "llama2" → "llama3.2" everywhere
  FIX-2: decide_strategy() — "and"/"or" no longer blindly triggers Cypher generation
          Now uses a smarter scoring system with NEGATIVE patterns (common words)
  FIX-3: format_context() — was checking "FOUND:" which doesn't exist in stats context,
          causing stats answers to always return "I couldn't find relevant information"
          Fixed: unified context marker + stats context now works
  FIX-4: generate_nlp_answer() — fixed broken context check (same root cause as FIX-3)
          Added model connection check before calling Ollama
  FIX-5: search_with_generated_cypher() — added fallback to keyword search if Cypher fails
          Previously just returned empty results with no recovery
"""

import time
import requests
from typing import Dict, Any
from config import Config, Neo4jConnection
from search_kg import KnowledgeGraphSearch
from cypher_generator import CypherGenerator, get_schema_description
from loguru import logger


class HybridKnowledgeGraphQA:
    """
    Hybrid system that intelligently chooses between:
    1. Pre-defined search functions (fast, safe, reliable)
    2. LLM-generated Cypher (flexible, handles complex filters)
    """

    # FIX-1: Centralized model name — change here once, applies everywhere
    LLM_MODEL = "llama3.2"
    OLLAMA_URL = "http://localhost:11434/api/generate"

    def __init__(self, llm_provider="ollama"):
        self.llm_provider = llm_provider
        self.searcher = KnowledgeGraphSearch()
        self.searcher.connect()
        self.cypher_generator = CypherGenerator(llm_provider)
        self.config = Config()
        logger.info(f"✓ Hybrid system initialized | model={self.LLM_MODEL}")

    def close(self):
        self.searcher.close()

    # ─────────────────────────────────────────────────────────
    # FIX-2: STRATEGY DECISION — smarter than keyword matching
    # ─────────────────────────────────────────────────────────

    def decide_strategy(self, question: str) -> str:
        """
        Decide which retrieval strategy to use — score-based, not naive keyword match.

        predefined → fast, safe, pre-written Cypher functions
        generate   → LLM writes Cypher on the fly (for complex/numeric filters)
        """
        import re as _re
        q = question.lower().strip()

        # ── Strong predefined signals (+2 each) ──
        predefined_strong = [
            "how many", "statistics", "count", "total",
            "most cited", "top papers", "top 5", "top 10",
            "from iit", "iit tirupati", "get stats", "overview",
        ]

        # ── Strong generate signals (+2 each) ──
        # Include typo variants like "more then" (common mistake for "more than")
        generate_strong = [
            "more than", "more then",        # typo variant
            "less than", "less then",        # typo variant
            "greater than", "fewer than",
            "at least", "at most",
            "published after", "published before",
            "between", "not from", "except", "excluding",
            "citations >", "citations <",
            "> 100", "> 500", "< 100",
        ]

        # ── Weak generate signals (+1 each) ──
        generate_weak = [
            "and also", "both", "who have written",
            "have more", "no more than",
        ]

        # Calculate scores
        score_predefined = sum(2 for p in predefined_strong if p in q)
        score_generate   = sum(2 for p in generate_strong   if p in q)
        score_generate  += sum(1 for p in generate_weak     if p in q)

        # Bonus: number + citation/year keyword → always needs Cypher filter
        has_number = bool(_re.search(r'\d+', q))
        citation_kw = any(kw in q for kw in ["citation", "cited", "cite"])
        year_kw     = any(kw in q for kw in ["year", "after", "before", "since"])
        if has_number and (citation_kw or year_kw):
            score_generate += 2
            logger.info("  → number + filter keyword → boosting generate score by 2")



        if score_predefined > score_generate:
            strategy = "predefined"
        elif score_generate >= 2:
            strategy = "generate"
        else:
            # Default: predefined is safer (won't hallucinate bad Cypher)
            strategy = "predefined"

        logger.info(
            f"Strategy: {strategy} "
            f"(predefined_score={score_predefined}, generate_score={score_generate})"
        )
        return strategy

    # ─────────────────────────────────────────────────────────
    # PREDEFINED SEARCH
    # ─────────────────────────────────────────────────────────

    def search_with_predefined(self, question: str) -> Dict[str, Any]:
        """Use pre-defined, tested search functions from search_kg.py"""
        q = question.lower()
        results = {"papers": [], "statistics": {}}

        if any(w in q for w in ["how many", "statistics", "count", "total", "overview"]):
            results["statistics"] = self.searcher.get_statistics()
            return results

        if any(w in q for w in ["most cited", "top papers", "best", "highest"]):
            results["papers"] = self.searcher.get_most_cited_papers(5)
            return results

        if "iit" in q or "institution" in q or "university" in q:
            results["papers"] = self.searcher.search_by_institution("IIT")
            return results

        if "author" in q:
            # Try to extract capitalized name from question
            words = question.split()
            for word in words:
                if word.istitle() and len(word) > 2 and word.lower() not in ["how", "what", "who", "the"]:
                    results["papers"] = self.searcher.search_by_author(word)
                    if results["papers"]:
                        return results

        # Default: keyword search on the full question
        results["papers"] = self.searcher.search_papers_by_keyword(question)
        return results

    # ─────────────────────────────────────────────────────────
    # FIX-5: GENERATED CYPHER SEARCH + FALLBACK
    # ─────────────────────────────────────────────────────────

    def search_with_generated_cypher(self, question: str) -> Dict[str, Any]:
        """
        Generate Cypher with LLM, execute it.
        FIX-5: If Cypher generation OR execution fails, fall back to keyword search
                instead of returning empty results.
        """
        logger.info("Generating Cypher query via LLM...")

        schema = get_schema_description()
        cypher_result = self.cypher_generator.generate_cypher(question, schema)

        if not cypher_result.get("cypher"):
            logger.warning("Cypher generation failed — falling back to keyword search")
            fallback = self.searcher.search_papers_by_keyword(question)
            return {
                "papers": fallback,
                "fallback": True,
                "fallback_reason": cypher_result.get("error", "generation failed"),
            }

        cypher_query = cypher_result["cypher"]
        logger.info(f"Executing: {cypher_query[:120]}...")

        try:
            with Neo4jConnection(self.config) as db:
                papers = db.execute_query(cypher_query)
                logger.info(f"✓ Cypher returned {len(papers)} results")
                return {
                    "papers": papers,
                    "cypher": cypher_query,
                    "explanation": cypher_result.get("explanation", ""),
                }

        except Exception as e:
            logger.error(f"Cypher execution failed: {e} — falling back to keyword search")
            # FIX-5: Fallback instead of empty results
            fallback = self.searcher.search_papers_by_keyword(question)
            return {
                "papers": fallback,
                "fallback": True,
                "fallback_reason": str(e),
                "failed_cypher": cypher_query,
            }

    # ─────────────────────────────────────────────────────────
    # FIX-3: FORMAT CONTEXT — fixed context marker bug
    # ─────────────────────────────────────────────────────────

    def format_context(self, search_results: Dict[str, Any]) -> str:
        """
        Format search results into a context string for the LLM answer generator.

        FIX-3 (OLD BUG): Context always started with "DATABASE STATISTICS:" or
          "RESEARCH PAPERS FOUND:" but generate_nlp_answer() checked for "FOUND:"
          which is NOT in "RESEARCH PAPERS FOUND:" — so stats-based answers always
          hit the early return "I couldn't find relevant information".

        FIX: Added a sentinel line "## CONTEXT_START ##" so the check is reliable.
        """
        context_parts = ["## CONTEXT_START ##"]
        has_content = False

        # ── Statistics block ──
        if search_results.get("statistics"):
            stats = search_results["statistics"]
            has_content = True
            context_parts.append("\nDATABASE STATISTICS:")
            context_parts.append(f"  Total Papers:       {stats.get('total_papers', 0)}")
            context_parts.append(f"  Total Authors:      {stats.get('total_authors', 0)}")
            context_parts.append(f"  Total Institutions: {stats.get('total_institutions', 0)}")
            context_parts.append(f"  Total Topics:       {stats.get('total_topics', 0)}")
            context_parts.append(
                f"  Total Citations:    {stats.get('total_citations', 0)}"
            )
            avg = stats.get("avg_citations", 0)
            context_parts.append(f"  Avg Citations/Paper: {avg:.1f}" if avg else "")

        # ── Papers block ──
        if search_results.get("papers"):
            has_content = True
            context_parts.append("\nRESEARCH PAPERS:")
            for i, paper in enumerate(search_results["papers"][:10], 1):
                title      = paper.get("title") or paper.get("p.title", "Untitled")
                year       = paper.get("year")  or paper.get("p.year", "")
                citations  = paper.get("citations") or paper.get("p.citation_count", 0)
                authors    = paper.get("authors", [])
                institution = paper.get("institution") or paper.get("institutions", [])

                context_parts.append(f"\n  {i}. {title}")
                if year:
                    context_parts.append(f"     Year: {year}")
                if citations is not None:
                    context_parts.append(f"     Citations: {citations}")
                if authors and isinstance(authors, list) and any(authors):
                    context_parts.append(f"     Authors: {', '.join(str(a) for a in authors[:3] if a)}")
                if institution:
                    if isinstance(institution, list):
                        inst_str = ", ".join(str(i) for i in institution if i)
                    else:
                        inst_str = str(institution)
                    if inst_str:
                        context_parts.append(f"     Institution: {inst_str}")

        # ── Fallback note ──
        if search_results.get("fallback"):
            context_parts.append(
                f"\n  [Note: Cypher failed ({search_results.get('fallback_reason','')}), "
                f"showing keyword-search results instead]"
            )

        if not has_content:
            return ""  # Caller will handle empty context

        return "\n".join(context_parts)

    # ─────────────────────────────────────────────────────────
    # FIX-4: GENERATE NLP ANSWER — fixed context check + model
    # ─────────────────────────────────────────────────────────

    def generate_nlp_answer(self, question: str, context: str) -> str:
        """
        Generate a natural language answer using Ollama LLM.

        FIX-4 (OLD BUG): Checked `"FOUND:" not in context` which was never True
          for statistics results → always returned "I couldn't find relevant information"
          even when stats data was present.

        FIX: Now checks for the "## CONTEXT_START ##" sentinel added in format_context().
        """
        # FIX-4: Correct sentinel check
        if not context or "## CONTEXT_START ##" not in context:
            return (
                "I couldn't find relevant information for your question in the knowledge graph. "
                "Try rephrasing, or check if the data has been ingested."
            )

        prompt = f"""You are a helpful research assistant. Answer the user's question using ONLY the data provided below.
Be specific. Cite paper titles, author names, and numbers from the data.
If the data doesn't fully answer the question, say so honestly.

DATA:
{context}

QUESTION: {question}

ANSWER:"""

        try:
            response = requests.post(
                self.OLLAMA_URL,
                json={
                    "model": self.LLM_MODEL,       # FIX-1: llama3.2
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.5,         # Slightly lower → more factual
                        "num_predict": 400,
                    }
                },
                timeout=90,
            )

            if response.status_code == 200:
                return response.json()["response"].strip()
            else:
                return f"LLM error: HTTP {response.status_code}"

        except requests.exceptions.ConnectionError:
            return "Error: Ollama is not running. Start it with: ollama serve"
        except requests.exceptions.Timeout:
            return "Error: LLM took too long to respond (>90s). Try a shorter question."
        except Exception as e:
            logger.error(f"LLM answer generation failed: {e}")
            return f"Error generating answer: {e}"

    # ─────────────────────────────────────────────────────────
    # MAIN ENTRY POINT
    # ─────────────────────────────────────────────────────────

    def ask(self, question: str) -> Dict[str, Any]:
        """
        Ask a question → get a natural language answer.

        Flow:
          1. decide_strategy()          → predefined or generate
          2. search_with_*()            → retrieve from Neo4j
          3. format_context()           → structure results for LLM
          4. generate_nlp_answer()      → LLM produces final answer
        """
        logger.info(f"Question: {question}")
        start_time = time.time()

        # Step 1
        strategy = self.decide_strategy(question)

        # Step 2
        if strategy == "predefined":
            search_results = self.search_with_predefined(question)
        else:
            search_results = self.search_with_generated_cypher(question)

        # Step 3
        context = self.format_context(search_results)

        # Step 4
        answer = self.generate_nlp_answer(question, context)

        elapsed = round(time.time() - start_time, 2)
        logger.info(f"✓ Done in {elapsed}s | strategy={strategy} | papers={len(search_results.get('papers', []))}")

        return {
            "question":     question,
            "strategy":     strategy,
            "answer":       answer,
            "context":      context,
            "papers_found": len(search_results.get("papers", [])),
            "latency_s":    elapsed,
            "fallback":     search_results.get("fallback", False),
        }


# ─────────────────────────────────────────────────────────
# INTERACTIVE MODE
# ─────────────────────────────────────────────────────────

def interactive_qa():
    print("\n" + "=" * 70)
    print("🤖  HYBRID KNOWLEDGE GRAPH Q&A")
    print("=" * 70)
    print(f"  Model     : llama3.2 (via Ollama)")
    print(f"  Retrieval : Predefined functions + LLM-generated Cypher")
    print(f"  Output    : Natural language answers")
    print("=" * 70)

    qa = HybridKnowledgeGraphQA("ollama")

    try:
        while True:
            print()
            question = input("🔍 Your question (or 'quit'): ").strip()

            if question.lower() in ["quit", "exit", "q"]:
                break
            if not question:
                continue

            print("\n⏳ Thinking...\n")
            result = qa.ask(question)

            print("=" * 70)
            print("💡 ANSWER:")
            print("=" * 70)
            print(result["answer"])
            print()
            print(f"📊 Strategy : {result['strategy']}")
            print(f"📄 Papers   : {result['papers_found']}")
            print(f"⏱  Latency  : {result['latency_s']}s")
            if result.get("fallback"):
                print("⚠️  Note     : Used keyword-search fallback (Cypher failed)")
            print("=" * 70)
    finally:
        qa.close()
        print("👋 Goodbye!")


if __name__ == "__main__":
    interactive_qa()