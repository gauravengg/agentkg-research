"""
RAG Evaluation System for Research Knowledge Graph Agent
=========================================================
Measures 4 key metrics:

  1. Tool Accuracy    — Did agent pick the RIGHT tool for the question?
  2. Answer Relevance — Does the answer actually address the question?
  3. Faithfulness     — Is the answer grounded in retrieved data (no hallucination)?
  4. Latency          — How fast does the agent respond?

Usage:
  python evaluator.py              # run full evaluation
  python evaluator.py --quick      # run first 5 questions only (for testing)
"""

from __future__ import annotations

import json
import time
import argparse
import requests
import re
from typing import Optional
from loguru import logger

# ── Only import agent if Neo4j is available ──
try:
    from agent import ResearchAgent
    AGENT_AVAILABLE = True
except Exception as e:
    logger.error(f"Could not import agent: {e}")
    AGENT_AVAILABLE = False

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL      = "llama3.2"


# ─────────────────────────────────────────────
# METRIC 1 — TOOL ACCURACY
# ─────────────────────────────────────────────

def compute_tool_accuracy(expected_tool: str, thought_chain: list) -> float:
    """
    Check if the agent used the expected tool at any step.

    Score:
      1.0 → correct tool used in Step 1 (best)
      0.5 → correct tool used in Step 2+ (found it eventually)
      0.0 → correct tool never used
    """
    if not thought_chain:
        return 0.0

    for step in thought_chain:
        used_tool = step.get("action", "")
        if used_tool == expected_tool:
            step_num = step.get("step", 99)
            return 1.0 if step_num == 1 else 0.5

    return 0.0


# ─────────────────────────────────────────────
# METRIC 2 — ANSWER RELEVANCE
# ─────────────────────────────────────────────

def compute_answer_relevance(question: str, answer: str) -> float:
    """
    Ask LLM: does this answer address the question?

    Score 0.0 to 1.0
    Uses simple keyword overlap as fallback if LLM fails.
    """
    if not answer or len(answer.strip()) < 5:
        return 0.0

    prompt = f"""Rate how well this answer addresses the question on a scale of 0 to 10.

QUESTION: {question}
ANSWER: {answer}

Scoring guide:
  10 = Perfectly answers the question with specific details
   7 = Mostly answers the question
   5 = Partially answers the question
   3 = Barely relevant
   0 = Completely irrelevant or says "I couldn't find"

Reply with ONLY a single number (0-10), nothing else."""

    raw = _call_llm(prompt, temperature=0.0, max_tokens=5)

    # Extract number from response
    match = re.search(r'\b(\d+(?:\.\d+)?)\b', raw.strip())
    if match:
        score = float(match.group(1))
        return min(score / 10.0, 1.0)

    # Fallback: keyword overlap between question and answer
    return _keyword_overlap(question, answer)


def _keyword_overlap(question: str, answer: str) -> float:
    """Simple fallback: what fraction of question keywords appear in answer"""
    stop_words = {"the", "a", "an", "is", "are", "in", "of", "to", "how",
                  "many", "what", "who", "show", "me", "there", "about"}
    q_words = {w.lower() for w in question.split() if w.lower() not in stop_words and len(w) > 2}
    a_lower = answer.lower()
    if not q_words:
        return 0.5
    matched = sum(1 for w in q_words if w in a_lower)
    return round(matched / len(q_words), 2)


# ─────────────────────────────────────────────
# METRIC 3 — FAITHFULNESS
# ─────────────────────────────────────────────

def compute_faithfulness(answer: str, context: str) -> float:
    """
    Ask LLM: is every claim in the answer supported by the context?

    Score 0.0 to 1.0
    Detects hallucinations — numbers/facts invented by LLM not in retrieved data.
    """
    if not answer or not context:
        return 0.0

    # Quick check: if answer says "couldn't find", it's not hallucinating
    if "couldn't find" in answer.lower() or "no results" in answer.lower():
        return 1.0   # Honest "no results" is faithful

    prompt = f"""Check if the ANSWER is supported by the CONTEXT data. Count claims that are supported vs not supported.

CONTEXT (retrieved data):
{context[:800]}

ANSWER to check:
{answer[:400]}

For each factual claim in the answer (numbers, names, titles, dates):
- Is it present in or directly derivable from the context?

Reply with ONLY this JSON:
{{"supported": <count of supported claims>, "total": <total claims>, "score": <supported/total>}}"""

    raw = _call_llm(prompt, temperature=0.0, max_tokens=100)

    # Try to parse JSON
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if match:
        try:
            data  = json.loads(match.group())
            score = float(data.get("score", 0))
            return min(max(score, 0.0), 1.0)
        except Exception:
            pass

    # Fallback: check if key numbers from context appear in answer
    numbers_in_context = set(re.findall(r'\b\d+\b', context))
    numbers_in_answer  = set(re.findall(r'\b\d+\b', answer))
    if not numbers_in_answer:
        return 0.8   # No numbers to verify, assume reasonable
    hallucinated = numbers_in_answer - numbers_in_context
    ratio = 1.0 - (len(hallucinated) / len(numbers_in_answer))
    return round(max(ratio, 0.0), 2)


# ─────────────────────────────────────────────
# MAIN EVALUATOR CLASS
# ─────────────────────────────────────────────

class AgentEvaluator:
    """
    Runs the full evaluation pipeline on eval_dataset.json
    and produces a report with all 4 metrics.
    """

    def __init__(self, dataset_path: str = "eval_dataset.json"):
        with open(dataset_path, "r") as f:
            self.dataset = json.load(f)
        logger.info(f"✓ Loaded {len(self.dataset)} evaluation questions")

        if AGENT_AVAILABLE:
            self.agent = ResearchAgent()
        else:
            self.agent = None
            logger.warning("Agent not available — will show metric structure only")

    def close(self):
        if self.agent:
            self.agent.close()

    def evaluate_single(self, item: dict) -> dict:
        """Run one question through the agent and score it on all metrics"""
        question     = item["question"]
        ground_truth = item["ground_truth"]
        expected_tool= item["expected_tool"]
        category     = item["category"]

        logger.info(f"\n[Q{item['id']}] {question}")

        start = time.time()

        if self.agent:
            result = self.agent.run(question)
            answer       = result.get("answer", "")
            thought_chain= result.get("thought_chain", [])
            context      = "\n".join(
                str(s.get("raw_results", "")) for s in thought_chain
            )
            latency = result.get("latency_s", round(time.time() - start, 2))
        else:
            answer        = "Agent not available"
            thought_chain = []
            context       = ""
            latency       = 0.0

        # ── Compute all metrics ──
        tool_accuracy    = compute_tool_accuracy(expected_tool, thought_chain)
        answer_relevance = compute_answer_relevance(question, answer)
        faithfulness     = compute_faithfulness(answer, context)

        # ── Ground truth similarity (simple keyword match) ──
        gt_similarity = _keyword_overlap(ground_truth, answer)

        eval_result = {
            "id"              : item["id"],
            "question"        : question,
            "category"        : category,
            "expected_tool"   : expected_tool,
            "answer"          : answer,
            "tool_accuracy"   : round(tool_accuracy, 2),
            "answer_relevance": round(answer_relevance, 2),
            "faithfulness"    : round(faithfulness, 2),
            "gt_similarity"   : round(gt_similarity, 2),
            "latency_s"       : latency,
            "steps_taken"     : len(thought_chain),
        }

        logger.info(
            f"  ✓ tool={tool_accuracy:.0%} | "
            f"relevance={answer_relevance:.0%} | "
            f"faith={faithfulness:.0%} | "
            f"latency={latency}s"
        )
        return eval_result

    def run_evaluation(self, quick: bool = False) -> dict:
        """
        Run evaluation on all (or first 5) questions.
        Returns full report with per-question results and aggregate scores.
        """
        dataset = self.dataset[:5] if quick else self.dataset
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting evaluation on {len(dataset)} questions")
        logger.info(f"{'='*60}")

        results      = []
        total_start  = time.time()

        for item in dataset:
            result = self.evaluate_single(item)
            results.append(result)

        total_time = round(time.time() - total_start, 2)

        # ── Aggregate scores ──
        def avg(key):
            vals = [r[key] for r in results if r[key] is not None]
            return round(sum(vals) / len(vals), 3) if vals else 0.0

        # Per-category breakdown
        categories = {}
        for r in results:
            cat = r["category"]
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(r)

        category_scores = {}
        for cat, items in categories.items():
            category_scores[cat] = {
                "count"           : len(items),
                "tool_accuracy"   : round(sum(i["tool_accuracy"] for i in items) / len(items), 2),
                "answer_relevance": round(sum(i["answer_relevance"] for i in items) / len(items), 2),
                "faithfulness"    : round(sum(i["faithfulness"] for i in items) / len(items), 2),
            }

        report = {
            "summary": {
                "total_questions"     : len(results),
                "avg_tool_accuracy"   : avg("tool_accuracy"),
                "avg_answer_relevance": avg("answer_relevance"),
                "avg_faithfulness"    : avg("faithfulness"),
                "avg_gt_similarity"   : avg("gt_similarity"),
                "avg_latency_s"       : avg("latency_s"),
                "avg_steps"           : avg("steps_taken"),
                "total_eval_time_s"   : total_time,
            },
            "by_category" : category_scores,
            "per_question": results,
        }

        return report

    def print_report(self, report: dict):
        """Print a clean evaluation report to terminal"""
        s = report["summary"]

        print("\n" + "=" * 65)
        print("📊  EVALUATION REPORT — Research Knowledge Graph Agent")
        print("=" * 65)

        print(f"\n{'METRIC':<30} {'SCORE':>10} {'MEANING'}")
        print("-" * 65)
        print(f"{'Tool Accuracy':<30} {s['avg_tool_accuracy']:>9.1%}  Did agent pick the right tool?")
        print(f"{'Answer Relevance':<30} {s['avg_answer_relevance']:>9.1%}  Does answer address the question?")
        print(f"{'Faithfulness':<30} {s['avg_faithfulness']:>9.1%}  No hallucination in answers?")
        print(f"{'Ground Truth Similarity':<30} {s['avg_gt_similarity']:>9.1%}  Matches expected answers?")
        print(f"{'Avg Latency':<30} {s['avg_latency_s']:>9.2f}s  Response time per question")
        print(f"{'Avg Steps':<30} {s['avg_steps']:>9.2f}   Tool calls per question")
        print(f"{'Questions Evaluated':<30} {s['total_questions']:>10}")

        print(f"\n{'='*65}")
        print("📂  SCORES BY CATEGORY")
        print(f"{'='*65}")
        print(f"{'Category':<18} {'Count':>5} {'Tool%':>7} {'Relevance':>10} {'Faithful':>10}")
        print("-" * 65)
        for cat, scores in report["by_category"].items():
            print(
                f"{cat:<18} {scores['count']:>5} "
                f"{scores['tool_accuracy']:>7.1%} "
                f"{scores['answer_relevance']:>10.1%} "
                f"{scores['faithfulness']:>10.1%}"
            )

        print(f"\n{'='*65}")
        print("📝  PER-QUESTION RESULTS")
        print(f"{'='*65}")
        print(f"{'ID':<4} {'Tool%':>5} {'Rel%':>5} {'Faith%':>7} {'Lat':>5}  Question")
        print("-" * 65)
        for r in report["per_question"]:
            print(
                f"{r['id']:<4} "
                f"{r['tool_accuracy']:>4.0%}  "
                f"{r['answer_relevance']:>4.0%}  "
                f"{r['faithfulness']:>5.0%}  "
                f"{r['latency_s']:>4.1f}s  "
                f"{r['question'][:40]}"
            )

        # Resume line
        print(f"\n{'='*65}")
        print("💼  RESUME LINE:")
        print(f"{'='*65}")
        ta  = s['avg_tool_accuracy']
        ar  = s['avg_answer_relevance']
        fa  = s['avg_faithfulness']
        n   = s['total_questions']
        print(
            f'\n  "Built an Agentic GraphRAG system evaluated on {n} benchmark questions,\n'
            f'   achieving {ta:.0%} tool accuracy, {ar:.0%} answer relevance,\n'
            f'   and {fa:.0%} faithfulness (hallucination-free rate)."\n'
        )
        print("=" * 65)

    def save_report(self, report: dict, path: str = "eval_report.json"):
        """Save full report to JSON for later reference"""
        with open(path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"✓ Report saved to {path}")


# ─────────────────────────────────────────────
# LLM HELPER (standalone, no agent needed)
# ─────────────────────────────────────────────

def _call_llm(prompt: str, temperature: float = 0.0, max_tokens: int = 100) -> str:
    """Call Ollama for metric computation"""
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model" : MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": temperature, "num_predict": max_tokens},
            },
            timeout=60,
        )
        if response.status_code == 200:
            return response.json()["response"]
        return ""
    except Exception as e:
        logger.warning(f"LLM call failed in evaluator: {e}")
        return ""


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the Research Agent")
    parser.add_argument("--quick", action="store_true", help="Run only first 5 questions")
    parser.add_argument("--save",  action="store_true", help="Save report to eval_report.json")
    args = parser.parse_args()

    evaluator = AgentEvaluator("eval_dataset.json")

    try:
        report = evaluator.run_evaluation(quick=args.quick)
        evaluator.print_report(report)

        if args.save:
            evaluator.save_report(report)

    finally:
        evaluator.close()