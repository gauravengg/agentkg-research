# AgentKG  
### Agentic GraphRAG Research Assistant

AgentKG is an **agent-based GraphRAG system** that combines a **Neo4j Knowledge Graph**, **LLaMA 3.2**, and a **ReAct-style agent** to enable intelligent, faithful, and explainable **research paper question answering**.

Unlike traditional RAG pipelines, AgentKG uses **tool-using agents** that reason over a structured graph and dynamically generate Cypher queries to retrieve precise, grounded information.

---

## 🔍 Key Features

- 🧠 ReAct Agent with explicit reasoning and tool selection  
- 🕸️ Neo4j Knowledge Graph for structured scholarly data  
- 🧾 LLM-generated Cypher queries for flexible KG access  
- 🔧 6 specialized graph tools for semantic + structural search  
- 📊 Evaluation pipeline measuring accuracy, relevance, and faithfulness  
- 🖥️ Streamlit-based interactive dashboard  

---

## 🏗️ System Architecture
User Question
↓
ReAct Agent (reason + choose tool)
↓
Neo4j Knowledge Graph
↓
LLaMA 3.2 (answer synthesis)
↓
Final Answer (grounded + faithful)

---

## 🧰 Agent Tools

The ReAct agent dynamically selects from the following tools:

- `keyword_search`
- `author_search`
- `institution_search`
- `topic_search`
- `cypher_search`
- `get_statistics`

Each tool operates directly on the knowledge graph, ensuring structured and verifiable retrieval.

---

## 🕸️ Knowledge Graph Schema
(Author)-[:WROTE]->(Paper)-[:ABOUT]->(Topic)
(Author)-[:AFFILIATED_WITH]->(Institution)
(Paper)-[:CITES]->(Paper)

This schema supports author-centric queries, topic discovery, citation analysis, and institution-level insights.

---

## 📊 Evaluation Results

Evaluation was performed on a **20-question benchmark** covering author, topic, institution, and citation-based queries.

| Metric           | Score |
|------------------|-------|
| Tool Accuracy    | ~85%  |
| Answer Relevance | ~78%  |
| Faithfulness     | ~90%  |
| Avg Latency      | ~6 s  |

The evaluation pipeline is inspired by **RAGAS-style metrics**, focusing on groundedness and correctness.

---

## 🚀 Quick Start

### 1️⃣ Clone & Install

```bash
git clone https://github.com/gauravengg/agentkg-research.git
cd agentkg-research
pip install -r requirements.txt

#########configure environment
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password

Run the System
python add_sample_data.py   # Load sample KG data
streamlit run app.py        # Launch Streamlit UI
python evaluator.py --quick # Run evaluation

├── agent.py               # ReAct agent logic
├── app.py                 # Streamlit dashboard
├── cypher_generator.py    # LLM → Cypher generation
├── evaluator.py           # Evaluation pipeline
├── eval_dataset.json      # 20-question benchmark
├── search_kg.py           # KG search utilities
├── config.py              # Neo4j configuration
└── requirements.txt




