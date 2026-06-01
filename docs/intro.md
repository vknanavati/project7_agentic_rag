# Project 7: Agentic RAG

## What This Project Is

A **birding question-answering agent** that combines the retrieval power of RAG (Project 5) with the reasoning loop of a ReAct Agent (Project 6). Instead of searching a knowledge base once and hoping for the best, this agent plans its searches, evaluates whether it has enough information, and keeps retrieving until it can fully answer the question.

The knowledge base is the same 22-species bird database built in Project 5 — no rebuilding needed. What's new is the intelligence layer on top: an agent that decides *how* to search, *what* to search for, and *when* it's done.

---

## The Problem With Single-Pass RAG

In Project 5, retrieval was a fixed pipeline:

```
User question → embed → search ChromaDB → rerank → send to LLM → answer
```

This works well for simple questions. It breaks down for complex ones.

Ask "Which backyard birds cache food, and how does that behavior compare between species?" and single-pass RAG will embed the full question, retrieve the top 5 chunks, and generate an answer from whatever it got — even if the chunks only covered one species, or missed the caching behavior entirely.

There is no second chance. No self-correction. No awareness that the question had two parts.

---

## What Agentic RAG Does Instead

The agent runs a reasoning loop — the same ReAct pattern from Project 6 — but its primary tools are retrieval functions over the bird knowledge base:

```
User question
    → Thought: "This has two parts. I'll search for food-caching birds first."
    → Action: search_knowledge_base("birds that cache food")
    → Observation: [chunks about Black-capped Chickadee, Red-breasted Nuthatch, Blue Jay]
    → Thought: "Good. Now I need the behavioral comparison between species."
    → Action: compare_species("Black-capped Chickadee", "Blue Jay")
    → Observation: [chunks comparing caching strategies]
    → Thought: "I now have enough to answer both parts fully."
    → Final Answer: ...
```

The agent plans before searching, evaluates what it retrieved, and decides whether to search again — all before committing to an answer.

---

## How This Builds on Previous Projects

| Project | What It Contributed |
|---|---|
| Project 5 | ChromaDB knowledge base, embeddings, BM25 search, re-ranking |
| Project 6 | ReAct reasoning loop, tool registry, LLM output parsing |
| **Project 7** | **Multi-tool retrieval agent, tool schemas, multi-hop reasoning, confidence-gated answers** |

This project does not rebuild the knowledge base or re-implement the ReAct loop from scratch. It integrates both, adds new retrieval tools, and introduces the intelligence to use them strategically.

---

## The Four Retrieval Tools

| Tool | When the Agent Uses It |
|---|---|
| `search_knowledge_base` | General semantic search — best for open-ended questions |
| `keyword_search` | BM25 keyword search — best when the question uses specific terms |
| `filter_by_species` | Pulls all chunks about one named bird — best for species-specific questions |
| `compare_species` | Pulls chunks about two birds side-by-side — best for comparison questions |

The agent reads descriptions of each tool and decides which one fits the question. It can call any combination, in any order, as many times as needed.

---

## The 7-Script Structure

| Script | Role |
|---|---|
| `config.py` | All constants: paths, model names, Groq settings, agent parameters |
| `knowledge_base.py` | Connects to the existing ChromaDB from Project 5 |
| `tools.py` | Defines the four retrieval tools and their schemas |
| `agent.py` | The ReAct reasoning loop: plan → retrieve → evaluate → repeat |
| `evaluate.py` | Benchmarks the agent on multi-hop questions vs. single-pass RAG |
| `predict.py` | CLI interface for interactive queries |
| `app.py` | Flask API on port 5004 |

---

## New Concepts in This Project

**Tool schemas** — Each tool is described to the LLM in structured text: what it does, what arguments it takes, what it returns, and when to use it. The LLM reads these descriptions at the start of every reasoning loop and uses them to decide which tool fits.

**Multi-hop reasoning** — Complex questions require chaining multiple retrievals. The agent recognizes when a question has multiple parts and plans its searches accordingly, rather than treating the whole question as a single embedding.

**Confidence-gated answers** — The agent explicitly checks before committing to a Final Answer: *"Have I retrieved enough to fully answer all parts of this question?"* If the answer is no, it searches again. This is the key behavior that separates an agentic system from a pipeline.

---

## Tech Stack

| Component | Tool |
|---|---|
| LLM | Groq API (`llama-3.1-8b-instant`) |
| Vector store | ChromaDB (reused from Project 5) |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| Keyword search | BM25 (rank-bm25) |
| Web framework | Flask |
| Language | Python 3.12.9 |

---

## What Success Looks Like

At the end of this project you will have:

- An agent that correctly answers multi-part birding questions that single-pass RAG gets wrong
- A benchmark showing the performance difference between agentic and single-pass retrieval
- A working Flask API that accepts questions and returns answers with citations
- A GitHub repository at `vknanavati/project7_agentic_rag`