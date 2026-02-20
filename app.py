"""
AgentKG — Research Intelligence Dashboard
Streamlit app with 3 tabs:
  1. Ask Agent   — ReAct agent Q&A with thought chain display
  2. Evaluation  — Run eval metrics with charts
  3. Graph Stats — KG overview with paper list
"""

import streamlit as st
import time
import json
import os

# ── Page config (must be first Streamlit call) ──
st.set_page_config(
    page_title="AgentKG — Research Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;500;600&display=swap');

/* Base */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Dark background */
.stApp {
    background: #0a0e1a;
    color: #e2e8f0;
}

/* Header */
.main-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
    border: 1px solid #334155;
    border-radius: 16px;
    padding: 32px 40px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.main-header::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(99,102,241,0.15) 0%, transparent 70%);
    pointer-events: none;
}
.main-title {
    font-family: 'Space Mono', monospace;
    font-size: 2.4rem;
    font-weight: 700;
    color: #e2e8f0;
    letter-spacing: -1px;
    margin: 0;
}
.main-title span { color: #818cf8; }
.main-subtitle {
    color: #94a3b8;
    font-size: 0.95rem;
    margin-top: 6px;
    font-weight: 300;
}

/* Metric cards */
.metric-card {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: #6366f1; }
.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: #818cf8;
}
.metric-label {
    font-size: 0.78rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 4px;
}

/* Thought chain steps */
.step-card {
    background: #0f172a;
    border-left: 3px solid #6366f1;
    border-radius: 0 10px 10px 0;
    padding: 16px 20px;
    margin-bottom: 12px;
}
.step-header {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    color: #6366f1;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 8px;
}
.step-thought {
    color: #94a3b8;
    font-size: 0.88rem;
    font-style: italic;
    margin-bottom: 6px;
}
.step-action {
    background: #1e293b;
    color: #22d3ee;
    font-family: 'Space Mono', monospace;
    font-size: 0.82rem;
    padding: 6px 12px;
    border-radius: 6px;
    display: inline-block;
    margin: 4px 0;
}
.step-obs {
    color: #a3e635;
    font-size: 0.83rem;
    margin-top: 6px;
}

/* Answer box */
.answer-box {
    background: linear-gradient(135deg, #0f172a, #1e1b4b);
    border: 1px solid #4f46e5;
    border-radius: 12px;
    padding: 24px 28px;
    color: #e2e8f0;
    font-size: 1rem;
    line-height: 1.7;
    margin-top: 8px;
}

/* Tags */
.tag {
    display: inline-block;
    background: #1e293b;
    color: #818cf8;
    font-size: 0.72rem;
    padding: 3px 10px;
    border-radius: 20px;
    margin: 2px;
    font-family: 'Space Mono', monospace;
}

/* Score bar */
.score-bar-wrap { margin: 6px 0; }
.score-bar-label {
    font-size: 0.78rem;
    color: #94a3b8;
    display: flex;
    justify-content: space-between;
    margin-bottom: 4px;
}
.score-bar-bg {
    background: #1e293b;
    border-radius: 4px;
    height: 8px;
    overflow: hidden;
}
.score-bar-fill {
    height: 100%;
    border-radius: 4px;
    background: linear-gradient(90deg, #6366f1, #818cf8);
    transition: width 0.6s ease;
}

/* Paper card */
.paper-card {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 10px;
    transition: border-color 0.2s;
}
.paper-card:hover { border-color: #4f46e5; }
.paper-title {
    font-size: 0.95rem;
    font-weight: 500;
    color: #e2e8f0;
    margin-bottom: 6px;
}
.paper-meta {
    font-size: 0.78rem;
    color: #64748b;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #0f172a;
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
    border: 1px solid #1e293b;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 8px;
    color: #64748b;
    font-family: 'Space Mono', monospace;
    font-size: 0.82rem;
    padding: 8px 20px;
}
.stTabs [aria-selected="true"] {
    background: #1e293b !important;
    color: #818cf8 !important;
}

/* Input */
.stTextInput input {
    background: #0f172a !important;
    border: 1px solid #334155 !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.9rem !important;
    padding: 14px 16px !important;
}
.stTextInput input:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 2px rgba(99,102,241,0.2) !important;
}

/* Button */
.stButton button {
    background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.85rem !important;
    padding: 10px 28px !important;
    transition: opacity 0.2s !important;
    width: 100% !important;
}
.stButton button:hover { opacity: 0.88 !important; }

/* Divider */
hr { border-color: #1e293b !important; }

/* Spinner */
.stSpinner { color: #818cf8 !important; }

/* Success/info */
.status-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #22d3ee;
    margin-right: 6px;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LOAD AGENT (cached)
# ─────────────────────────────────────────────

@st.cache_resource
def load_agent():
    try:
        from agent import ResearchAgent
        return ResearchAgent(), None
    except Exception as e:
        return None, str(e)


@st.cache_resource
def load_searcher():
    try:
        from search_kg import KnowledgeGraphSearch
        s = KnowledgeGraphSearch()
        s.connect()
        return s, None
    except Exception as e:
        return None, str(e)


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────

st.markdown("""
<div class="main-header">
    <div class="main-title">Agent<span>KG</span></div>
    <div class="main-subtitle">
        Agentic GraphRAG · Neo4j Knowledge Graph · ReAct Agent · LLaMA 3.2
    </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(["🔍  Ask Agent", "📊  Evaluation", "🗄️  Graph Stats"])


# ═══════════════════════════════════════════════
# TAB 1 — ASK AGENT
# ═══════════════════════════════════════════════

with tab1:
    agent, agent_err = load_agent()

    if agent_err:
        st.error(f"Could not load agent: {agent_err}")
    else:
        st.markdown("<br>", unsafe_allow_html=True)

        # Example questions
        st.markdown('<div style="color:#64748b;font-size:0.78rem;font-family:Space Mono,monospace;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px">Try asking</div>', unsafe_allow_html=True)
        examples = [
            "how many papers are in the database",
            "papers with more than 100 citations",
            "papers from IIT Tirupati",
            "papers about deep learning",
            "papers published after 2020",
        ]
        cols = st.columns(len(examples))
        for i, (col, ex) in enumerate(zip(cols, examples)):
            with col:
                if st.button(ex, key=f"ex_{i}"):
                    st.session_state["question_input"] = ex

        st.markdown("<br>", unsafe_allow_html=True)

        # Input
        col_input, col_btn = st.columns([5, 1])
        with col_input:
            question = st.text_input(
                "Question",
                placeholder="Ask anything about the research corpus...",
                key="question_input",
                label_visibility="collapsed",
            )
        with col_btn:
            ask_btn = st.button("Ask →")

        st.markdown("<hr>", unsafe_allow_html=True)

        # Run agent
        if ask_btn and question:
            with st.spinner("Agent reasoning..."):
                start = time.time()
                result = agent.run(question)
                elapsed = time.time() - start

            thought_chain = result.get("thought_chain", [])
            answer        = result.get("answer", "No answer generated")
            steps         = result.get("steps_taken", 0)

            # Metrics row
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(f'<div class="metric-card"><div class="metric-value">{steps}</div><div class="metric-label">Steps Taken</div></div>', unsafe_allow_html=True)
            with c2:
                st.markdown(f'<div class="metric-card"><div class="metric-value">{elapsed:.1f}s</div><div class="metric-label">Latency</div></div>', unsafe_allow_html=True)
            with c3:
                papers_found = result.get("papers_found", len(thought_chain))
                st.markdown(f'<div class="metric-card"><div class="metric-value">{result.get("steps_taken",0)}</div><div class="metric-label">Tool Calls</div></div>', unsafe_allow_html=True)
            with c4:
                success_icon = "✅" if result.get("success") else "⚠️"
                st.markdown(f'<div class="metric-card"><div class="metric-value">{success_icon}</div><div class="metric-label">Status</div></div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Thought chain
            if thought_chain:
                st.markdown('<div style="color:#64748b;font-size:0.78rem;font-family:Space Mono,monospace;text-transform:uppercase;letter-spacing:1px;margin-bottom:12px">🧠 Agent Thought Chain</div>', unsafe_allow_html=True)
                for step in thought_chain:
                    thought  = step.get("thought", "")
                    action   = step.get("action", "")
                    args     = step.get("args", {})
                    obs      = step.get("observation", "")[:300]

                    st.markdown(f"""
                    <div class="step-card">
                        <div class="step-header">Step {step.get('step', '?')}</div>
                        <div class="step-thought">💭 {thought}</div>
                        <div class="step-action">🔧 {action}({json.dumps(args)})</div>
                        <div class="step-obs">👁 {obs}</div>
                    </div>
                    """, unsafe_allow_html=True)

            # Final answer
            st.markdown('<div style="color:#64748b;font-size:0.78rem;font-family:Space Mono,monospace;text-transform:uppercase;letter-spacing:1px;margin:16px 0 8px">💡 Final Answer</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)

        elif not question and ask_btn:
            st.warning("Please enter a question.")


# ═══════════════════════════════════════════════
# TAB 2 — EVALUATION
# ═══════════════════════════════════════════════

with tab2:
    st.markdown("<br>", unsafe_allow_html=True)

    col_left, col_right = st.columns([2, 1])

    with col_right:
        st.markdown("""
        <div style="background:#0f172a;border:1px solid #1e293b;border-radius:12px;padding:20px">
            <div style="font-family:'Space Mono',monospace;font-size:0.78rem;color:#64748b;text-transform:uppercase;letter-spacing:1px;margin-bottom:12px">About Metrics</div>
            <div style="font-size:0.82rem;color:#94a3b8;line-height:1.8">
                <b style="color:#818cf8">Tool Accuracy</b><br>
                Did agent pick the right tool?<br><br>
                <b style="color:#818cf8">Answer Relevance</b><br>
                Does answer address the question?<br><br>
                <b style="color:#818cf8">Faithfulness</b><br>
                No hallucination — only uses retrieved data<br><br>
                <b style="color:#818cf8">GT Similarity</b><br>
                Matches expected ground truth answer
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_left:
        quick_mode = st.checkbox("Quick mode (5 questions only)", value=True)
        run_eval   = st.button("▶  Run Evaluation")

    if run_eval:
        try:
            from evaluator import AgentEvaluator

            with st.spinner(f"Running evaluation on {'5' if quick_mode else '20'} questions... (~{1 if quick_mode else 10} min)"):
                evaluator = AgentEvaluator("eval_dataset.json")
                report    = evaluator.run_evaluation(quick=quick_mode)
                evaluator.close()

            s = report["summary"]

            # Score cards
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div style="color:#64748b;font-size:0.78rem;font-family:Space Mono,monospace;text-transform:uppercase;letter-spacing:1px;margin-bottom:12px">Overall Scores</div>', unsafe_allow_html=True)

            c1, c2, c3, c4 = st.columns(4)
            metrics = [
                (c1, "Tool Accuracy",    s["avg_tool_accuracy"],    "#818cf8"),
                (c2, "Answer Relevance", s["avg_answer_relevance"], "#22d3ee"),
                (c3, "Faithfulness",     s["avg_faithfulness"],     "#a3e635"),
                (c4, "GT Similarity",    s["avg_gt_similarity"],    "#f472b6"),
            ]
            for col, label, val, color in metrics:
                with col:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value" style="color:{color}">{val:.0%}</div>
                        <div class="metric-label">{label}</div>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Category breakdown
            st.markdown('<div style="color:#64748b;font-size:0.78rem;font-family:Space Mono,monospace;text-transform:uppercase;letter-spacing:1px;margin-bottom:12px">By Category</div>', unsafe_allow_html=True)

            for cat, scores in report["by_category"].items():
                with st.expander(f"📂  {cat.upper()}  ({scores['count']} questions)"):
                    for metric_key, label in [
                        ("tool_accuracy",    "Tool Accuracy"),
                        ("answer_relevance", "Answer Relevance"),
                        ("faithfulness",     "Faithfulness"),
                    ]:
                        val = scores[metric_key]
                        st.markdown(f"""
                        <div class="score-bar-wrap">
                            <div class="score-bar-label">
                                <span>{label}</span><span>{val:.0%}</span>
                            </div>
                            <div class="score-bar-bg">
                                <div class="score-bar-fill" style="width:{val*100:.0f}%"></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Per question table
            st.markdown('<div style="color:#64748b;font-size:0.78rem;font-family:Space Mono,monospace;text-transform:uppercase;letter-spacing:1px;margin-bottom:12px">Per Question Results</div>', unsafe_allow_html=True)

            for r in report["per_question"]:
                ta  = r["tool_accuracy"]
                ar  = r["answer_relevance"]
                fa  = r["faithfulness"]
                dot = "🟢" if ta == 1.0 else ("🟡" if ta >= 0.5 else "🔴")

                with st.expander(f"{dot}  Q{r['id']}: {r['question'][:55]}"):
                    col_a, col_b = st.columns([3, 2])
                    with col_a:
                        st.markdown(f"**Answer:** {r['answer'][:300]}")
                    with col_b:
                        st.markdown(f"""
                        <div style="font-size:0.82rem;color:#94a3b8;line-height:2">
                            Tool used: <span style="color:#22d3ee;font-family:Space Mono,monospace">{r.get('expected_tool','?')}</span><br>
                            Tool accuracy: <b style="color:#818cf8">{ta:.0%}</b><br>
                            Relevance: <b style="color:#22d3ee">{ar:.0%}</b><br>
                            Faithfulness: <b style="color:#a3e635">{fa:.0%}</b><br>
                            Latency: <b style="color:#f472b6">{r['latency_s']:.1f}s</b><br>
                            Steps: <b style="color:#e2e8f0">{r['steps_taken']}</b>
                        </div>
                        """, unsafe_allow_html=True)

            # Resume line
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,#0f172a,#1e1b4b);border:1px solid #4f46e5;border-radius:12px;padding:24px 28px">
                <div style="font-family:'Space Mono',monospace;font-size:0.75rem;color:#6366f1;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:12px">💼 Your Resume Line</div>
                <div style="color:#e2e8f0;font-size:0.95rem;line-height:1.8">
                    "Built an Agentic GraphRAG system evaluated on <b style="color:#818cf8">{s['total_questions']} benchmark questions</b>, 
                    achieving <b style="color:#818cf8">{s['avg_tool_accuracy']:.0%} tool accuracy</b>, 
                    <b style="color:#22d3ee">{s['avg_answer_relevance']:.0%} answer relevance</b>, 
                    and <b style="color:#a3e635">{s['avg_faithfulness']:.0%} faithfulness</b> (hallucination-free rate)."
                </div>
            </div>
            """, unsafe_allow_html=True)

        except FileNotFoundError:
            st.error("eval_dataset.json not found. Make sure it's in the same folder as app.py")
        except Exception as e:
            st.error(f"Evaluation error: {e}")


# ═══════════════════════════════════════════════
# TAB 3 — GRAPH STATS
# ═══════════════════════════════════════════════

with tab3:
    searcher, search_err = load_searcher()
    st.markdown("<br>", unsafe_allow_html=True)

    if search_err:
        st.error(f"Could not connect to Neo4j: {search_err}")
    else:
        # Stats
        try:
            stats = searcher.get_statistics()

            c1, c2, c3, c4, c5 = st.columns(5)
            stat_items = [
                (c1, "Papers",       stats.get("total_papers", 0),        "#818cf8"),
                (c2, "Authors",      stats.get("total_authors", 0),        "#22d3ee"),
                (c3, "Topics",       stats.get("total_topics", 0),         "#a3e635"),
                (c4, "Institutions", stats.get("total_institutions", 0),   "#f472b6"),
                (c5, "Avg Citations",f"{stats.get('avg_citations') or 0:.0f}", "#fb923c"),
            ]
            for col, label, val, color in stat_items:
                with col:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value" style="color:{color}">{val}</div>
                        <div class="metric-label">{label}</div>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
        except Exception as e:
            st.warning(f"Could not load stats: {e}")

        # Top papers
        st.markdown('<div style="color:#64748b;font-size:0.78rem;font-family:Space Mono,monospace;text-transform:uppercase;letter-spacing:1px;margin-bottom:12px">📄 Top Papers by Citations</div>', unsafe_allow_html=True)

        try:
            papers = searcher.get_most_cited_papers(10)
            for p in papers:
                title     = p.get("title", "Untitled")
                year      = p.get("year", "")
                citations = p.get("citations", 0)
                authors   = p.get("authors", [])
                topics    = p.get("topics", [])

                auth_str  = ", ".join(a for a in authors[:2] if a) if authors else "Unknown"
                topic_tags= "".join(f'<span class="tag">{t}</span>' for t in topics[:3] if t)

                st.markdown(f"""
                <div class="paper-card">
                    <div class="paper-title">{title}</div>
                    <div class="paper-meta">
                        👤 {auth_str} &nbsp;·&nbsp;
                        📅 {year} &nbsp;·&nbsp;
                        📈 {citations:,} citations
                    </div>
                    <div style="margin-top:8px">{topic_tags}</div>
                </div>
                """, unsafe_allow_html=True)

        except Exception as e:
            st.warning(f"Could not load papers: {e}")

        st.markdown("<br>", unsafe_allow_html=True)

        # Custom search
        st.markdown('<div style="color:#64748b;font-size:0.78rem;font-family:Space Mono,monospace;text-transform:uppercase;letter-spacing:1px;margin-bottom:12px">🔎 Search Papers</div>', unsafe_allow_html=True)

        search_q = st.text_input("Search", placeholder="keyword, author, or topic...", key="kg_search", label_visibility="collapsed")
        if search_q:
            try:
                results = searcher.search_papers_by_keyword(search_q)
                if results:
                    st.markdown(f'<div style="color:#64748b;font-size:0.8rem;margin-bottom:10px">Found {len(results)} papers</div>', unsafe_allow_html=True)
                    for p in results[:8]:
                        title     = p.get("title", "Untitled")
                        citations = p.get("citations", 0)
                        year      = p.get("year", "")
                        authors   = p.get("authors", [])
                        auth_str  = ", ".join(a for a in authors[:2] if a) if authors else "Unknown"
                        st.markdown(f"""
                        <div class="paper-card">
                            <div class="paper-title">{title}</div>
                            <div class="paper-meta">👤 {auth_str} · 📅 {year} · 📈 {citations:,} citations</div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No papers found for that query.")
            except Exception as e:
                st.warning(f"Search error: {e}")


# ── Footer ──
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center;color:#334155;font-size:0.75rem;font-family:Space Mono,monospace;padding:20px;border-top:1px solid #1e293b">
    AgentKG · Neo4j + LLaMA 3.2 + ReAct · Agentic GraphRAG System
</div>
""", unsafe_allow_html=True)