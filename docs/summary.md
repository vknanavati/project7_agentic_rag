# Project 7 Summary — Agentic RAG

## What Was Built

A birding question-answering agent that combines the retrieval infrastructure from Project 5 with the reasoning loop from Project 6. The agent searches a 21-species bird knowledge base using four different retrieval strategies, evaluates what it finds, and keeps searching until it can fully answer the question or runs out of attempts.

---

## How It Differs From Prior Projects

**vs Project 5 (RAG)** — Project 5 searched once and stopped. The pipeline was fixed — embed the question, retrieve 5 chunks, send to LLM, done. No ability to recognize gaps or search again. Project 7 wraps that retrieval in a reasoning loop that decides how many searches to run and which tool to use on each one.

**vs Project 6 (ReAct Agent)** — Project 6 had the same agentic architecture but used general-purpose tools — a calculator, dictionary, and Wikipedia. The questions were simple enough that one tool call usually sufficed. Project 7 uses four specialized retrieval tools over a domain-specific knowledge base, and the questions are complex enough that the agent genuinely has to use multiple searches to answer them completely.

---

## What Agentic Means in This Project

Three properties make the system agentic:

**Reasoning loop** — The agent runs in a `while` loop, calling the LLM multiple times on the same question. Each call receives the full conversation history including every prior tool result. The loop only stops when the LLM writes Final Answer or hits the iteration limit.

**Tools** — Four Python functions that actually query ChromaDB and return real retrieved text. The LLM decides which tool to call by reading plain-English descriptions in the system prompt. The tool name gets written as text, our Python code runs the actual function, and the result gets appended to the conversation history.

**Self-evaluation** — Before every action the LLM writes a Thought assessing what it knows and what it still needs. After 4 searches it receives an explicit prompt to commit to a Final Answer using whatever evidence it has collected. The combination of the Thought structure and the forced commit instruction is what makes the agent stop at the right time rather than looping forever.

---

## Key Findings From Evaluation

Running 6 multi-hop questions through both the agent and single-pass RAG produced one clear finding: single-pass RAG performs adequately on focused single-species questions where one search covers everything needed. The agent earns its complexity on questions that require synthesizing information across multiple species or multiple topics — questions where a single search retrieval is structurally unable to cover all parts.

Both systems honestly acknowledged knowledge base gaps rather than hallucinating. This was a deliberate design choice in the system prompt — the agent is explicitly instructed to note when the knowledge base does not contain enough information rather than filling gaps with general knowledge.

---

## Technical Challenges

**Groq rate limits** — The 6,000 token per minute free tier limit required three separate mitigations: trimming retrieved chunks to 200 characters, trimming conversation history after 8 messages, and adding 15 second sleeps between iterations. The evaluation script takes 10-15 minutes as a result.

**LLM format compliance** — The LLM repeatedly violated the one-action-per-response rule by writing multiple tool calls in a single response, or writing `Action: None` instead of `Final Answer:`. Both required prompt fixes and a code-level None detection fallback in the reasoning loop.

**Knowledge base gaps** — Several evaluation questions exposed limits of the Project 5 knowledge base. Questions about feeder aggression, nesting materials, and bird intelligence hit topics the 21 species documents did not cover in depth. These are honest gaps — the agent correctly identified and reported them rather than fabricating answers.

---

## Curriculum Position

| Project | Concept | What It Added |
|---|---|---|
| 0 | CNN Image Classifier | Computer vision, PyTorch |
| 1 | Binary Classification | Tabular ML, scikit-learn |
| 2 | Sentiment Analysis | NLP, TF-IDF |
| 3 | Regression | XGBoost, feature engineering |
| 4 | Text Generation | RNN, LSTM, Transformer |
| 5 | RAG | ChromaDB, embeddings, hybrid search |
| 6 | ReAct Agent | Reasoning loop, tools, self-evaluation |
| **7** | **Agentic RAG** | **Multi-tool retrieval, domain-specific agent** |

Project 7 is the first project that combines two prior projects directly — the knowledge base infrastructure from Project 5 and the agentic architecture from Project 6. It demonstrates that agentic behavior is not a different kind of system but a reasoning layer placed on top of existing retrieval infrastructure.