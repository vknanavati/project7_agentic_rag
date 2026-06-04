# Project 7: Agentic RAG — Birding Assistant

## Overview

An agentic question-answering system built on top of the bird knowledge base from Project 5. Instead of searching once and hoping for the best, this system uses a ReAct reasoning loop to plan searches, evaluate retrieved evidence, and keep searching until it can fully answer the question.

Built with Groq (`llama-3.1-8b-instant`), ChromaDB, and Flask. Reuses the 21-species knowledge base built in Project 5 without rebuilding it.

---

## How It Works

The agent runs a reasoning loop on every question:

```
Thought → Action → Observation → Thought → Action → Observation → ... → Final Answer
```

On each iteration the agent reads the full conversation history — including every prior tool result — and decides whether to search again or commit to a Final Answer. It only stops when it has enough evidence to fully answer the question, or when it hits the maximum iteration limit.

---

## The Four Retrieval Tools

| Tool | Strategy | Best For |
|---|---|---|
| `search_knowledge_base` | Semantic vector search | Open-ended conceptual questions |
| `keyword_search` | BM25 keyword matching | Specific terms or species names |
| `filter_by_species` | Metadata filter by species tag | Deep single-species questions |
| `compare_species` | Pulls two species side by side | Explicit comparison questions |

---

## Project Structure

| Script | Role |
|---|---|
| `config.py` | All constants — paths, model names, agent parameters, Flask settings |
| `knowledge_base.py` | Connects to the existing ChromaDB from Project 5 |
| `tools.py` | Defines the four retrieval tools and their schemas |
| `agent.py` | The ReAct reasoning loop: Thought → Action → Observation → repeat |
| `evaluate.py` | Benchmarks the agent against single-pass RAG on 6 multi-hop questions |
| `predict.py` | CLI interface for interactive queries |
| `app.py` | Flask API on port 5004 |

---

## Setup

### Prerequisites

- Python 3.12
- Project 5 (`project5_birding_rag`) must exist at the same level as this project — the ChromaDB knowledge base is read directly from it
- A Groq API key from [console.groq.com](https://console.groq.com)

### Installation

```bash
cd ~/Programming/project7_agentic_rag
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Set the Groq API Key

```bash
export GROQ_API_KEY="your_groq_api_key_here"
```

---

## Usage

### CLI — Interactive Queries

```bash
python predict.py
```

Type questions at the prompt. Type `verbose` to toggle iteration-level output. Type `quit` to exit.

### Flask API

```bash
python app.py
```

The server starts on port 5004.

**Endpoints:**

```bash
# Health check
curl http://localhost:5004/health

# Ask a question
curl -X POST http://localhost:5004/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Which birds cache food for winter?"}'

# List all species
curl http://localhost:5004/species

# Compare two species
curl -X POST http://localhost:5004/compare \
  -H "Content-Type: application/json" \
  -d '{"species_a": "blue_jay", "species_b": "american_crow"}'
```

### Evaluation

```bash
python evaluate.py
```

Runs 6 multi-hop questions through both the agent and single-pass RAG and prints answers side by side with iteration counts. Takes approximately 10-15 minutes due to Groq rate limit pauses.

---

## Evaluation Results

| Question | Agent Iterations | Single-Pass Adequate? |
|---|---|---|
| Which birds cache food for winter, and how does behavior differ? | 5 | Partial — missed species comparison |
| What do Juncos and Chickadees eat, and how do feeder preferences differ? | 5 | Partial — less structured |
| Which birds are cavity nesters, and what nesting materials do they prefer? | 5 | Partial — missed nesting materials gap |
| How do American Goldfinches change appearance across seasons? | 5 | Yes — well covered in one search |
| Year-round residents vs seasonal visitors and winter feeder activity? | 5 | Partial — missed feeder management advice |
| How do Blue Jays and Crows communicate, and how intelligent are they? | 5 | Yes — knowledge base gap for both |

**Key finding:** Single-pass RAG performs adequately on focused single-species questions. The agent earns its complexity on multi-part questions that require synthesizing information across multiple species or topics. Both systems honestly acknowledged knowledge base gaps rather than hallucinating.

---

## Tech Stack

| Component | Tool |
|---|---|
| LLM | Groq API (`llama-3.1-8b-instant`) |
| Vector store | ChromaDB (reused from Project 5) |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| Keyword search | BM25 (`rank-bm25`) |
| Web framework | Flask |
| Language | Python 3.12 |

---

## How This Builds on Previous Projects

| Project | Contribution to This Project |
|---|---|
| Project 5 | ChromaDB knowledge base, embeddings, BM25 search |
| Project 6 | ReAct reasoning loop, tool registry, LLM output parsing |
| Project 7 | Multi-tool retrieval agent, tool schemas, confidence-gated answers |

---

## Known Limitations

- **Groq free tier rate limits** — 6,000 tokens per minute. The agent includes sleep pauses between iterations to stay within limits. The evaluation script takes 10-15 minutes as a result.
- **Knowledge base gaps** — The 21-species knowledge base was built around feeder tips and identification. Questions about intelligence, communication, and nesting materials hit the limits of what was documented in Project 5.
- **Forced Final Answer at iteration 4** — To prevent infinite loops on unanswerable questions, the agent is prompted to commit after 4 searches. On some questions this means the answer is incomplete rather than absent.