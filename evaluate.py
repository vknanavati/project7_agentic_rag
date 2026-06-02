# evaluate.py
# Benchmarks the Agentic RAG agent against single-pass RAG on multi-hop questions
# Multi-hop questions require chaining multiple retrievals to answer fully
# Single-pass RAG searches once and stops — the agent searches until satisfied
# Analogy: comparing a student who reads one chapter before an exam
#          versus one who keeps reading until they feel fully prepared

import time                                     # for sleeping between API calls to avoid rate limits
import os                                       # for environment variables
from groq import Groq                           # the Groq API client

import config                                   # our central settings file
from knowledge_base import load_knowledge_base  # connects to ChromaDB
from tools import search_knowledge_base         # single search tool for the baseline
from agent import run_agent                     # the full agentic reasoning loop


# --- Groq Client ---
# Instantiated once at module load time — same pattern as agent.py
client = Groq()


# --- Evaluation Questions ---
# These are deliberately multi-hop questions — each one has multiple parts
# that require at least two separate retrievals to answer fully
# Single-pass RAG will struggle because one search cannot cover all parts
# The agent should outperform it by searching multiple times
EVAL_QUESTIONS = [
    "Which backyard birds cache food for winter, and how does the caching behavior differ between species?",
    "What do Dark-eyed Juncos and Black-capped Chickadees eat, and which one is more aggressive at feeders?",
    "Which birds in the knowledge base are cavity nesters, and what kind of nesting materials do they prefer?",
    "How do American Goldfinches change appearance across seasons, and what foods attract them to feeders?",
    "Which birds are year-round residents versus seasonal visitors, and how does that affect feeder activity in winter?",
    "What is the difference between how Blue Jays and American Crows communicate, and how intelligent are each of them?",
]


# --- Single-Pass RAG Baseline ---

def run_single_pass_rag(question, collection, embedding_model):
    # Plain explanation: performs one semantic search on the question, sends the
    #                    retrieved chunks to the Groq LLM, and returns the answer
    #                    without any reasoning loop or self-evaluation
    # Analogy: a student who reads the first relevant chapter they find
    #          and writes their answer immediately without checking
    #          whether they covered all parts of the question

    # Search the knowledge base once with the full question as the query
    results = search_knowledge_base(question, collection, embedding_model)

    # Build a context string from the retrieved chunks
    # This is everything the LLM will have access to when generating its answer
    context_parts = []
    for i, result in enumerate(results, 1):                         # loop through retrieved chunks
        species = result["metadata"].get("species", "Unknown")      # get species name from metadata
        context_parts.append(f"[Chunk {i}] Species: {species}")     # label each chunk
        context_parts.append(result["text"])                        # add the chunk text
        context_parts.append("")                                    # blank line between chunks

    context = "\n".join(context_parts)                              # join into one string

    # Build the prompt for the LLM
    # This is a standard RAG prompt — context first, then question
    prompt = f"""You are a birding expert. Use the following retrieved chunks from a bird knowledge base to answer the question. Only use information from the chunks provided.

Retrieved Chunks:
{context}

Question: {question}

Answer:"""

    # Send to Groq and get the response — one call, no loop
    response = client.chat.completions.create(
        model=config.GROQ_MODEL,                                    # which LLM to use
        messages=[{"role": "user", "content": prompt}],            # single message, no history
        max_tokens=config.GROQ_MAX_TOKENS,                          # cap the response length
        temperature=config.GROQ_TEMPERATURE,                        # deterministic output
    )

    # Extract and return the answer text
    return response.choices[0].message.content.strip()


# --- Evaluation Runner ---

def run_evaluation(collection, embedding_model):
    # Plain explanation: loops through every evaluation question, runs both the
    #                    single-pass RAG baseline and the agentic system, prints
    #                    both answers side by side, and tracks summary statistics
    # Analogy: giving the same exam to two students and grading their answers
    #          side by side to see who prepared more thoroughly

    print("\n" + "="*70)
    print("EVALUATION: Agentic RAG vs Single-Pass RAG")
    print("="*70)

    # Track results for the summary at the end
    results_log = []                                                # stores each question's results

    for i, question in enumerate(EVAL_QUESTIONS, 1):               # loop through all eval questions

        print(f"\n{'='*70}")
        print(f"Question {i} of {len(EVAL_QUESTIONS)}:")
        print(f"{question}")
        print(f"{'='*70}")

        # --- Run Single-Pass RAG ---
        print("\n[Single-Pass RAG] Searching once...")
        single_pass_answer = run_single_pass_rag(question, collection, embedding_model)
        print(f"\nSingle-Pass Answer:\n{single_pass_answer}")

        # Sleep to avoid hitting Groq's rate limit between the two calls
        # Groq's free tier allows 6,000 tokens per minute
        print(f"\nWaiting {config.EVAL_SLEEP_SECONDS}s before running agent...")
        time.sleep(config.EVAL_SLEEP_SECONDS)                       # pause between API calls

        # --- Run Agentic RAG ---
        print("\n[Agentic RAG] Running reasoning loop...")
        agent_result = run_agent(                                   # run the full agent
            question=question,
            collection=collection,
            embedding_model=embedding_model,
            verbose=True,                                           # print each iteration as it runs
        )

        print(f"\nAgent Answer:\n{agent_result['answer']}")
        print(f"Iterations used: {agent_result['iterations']} of {config.MAX_ITERATIONS}")

        # Store the result for the summary
        results_log.append({
            "question": question,                                   # the question asked
            "single_pass_answer": single_pass_answer,               # what single-pass RAG said
            "agent_answer": agent_result["answer"],                 # what the agent said
            "iterations": agent_result["iterations"],               # how many loops the agent ran
        })

        # Sleep between questions to stay within Groq's rate limit
        if i < len(EVAL_QUESTIONS):                                 # no need to sleep after the last question
            print(f"\nWaiting {config.EVAL_SLEEP_SECONDS}s before next question...")
            time.sleep(config.EVAL_SLEEP_SECONDS)

    # --- Print Summary ---
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)

    # Calculate the average number of iterations the agent used
    total_iterations = sum(r["iterations"] for r in results_log)   # sum all iteration counts
    avg_iterations = total_iterations / len(results_log)           # compute the average

    print(f"Questions evaluated:       {len(EVAL_QUESTIONS)}")
    print(f"Total agent iterations:    {total_iterations}")
    print(f"Average iterations/question: {avg_iterations:.1f}")
    print(f"Max allowed iterations:    {config.MAX_ITERATIONS}")

    # Print a compact per-question iteration breakdown
    print(f"\nPer-question iteration counts:")
    for i, result in enumerate(results_log, 1):
        print(f"  Q{i}: {result['iterations']} iteration(s) — {result['question'][:60]}...")

    return results_log                                              # return for any further processing


# --- Entry Point ---
# Runs when you execute `python evaluate.py` directly
# Skipped when other scripts import from this file
if __name__ == "__main__":

    print("Loading knowledge base...")
    collection, embedding_model = load_knowledge_base()             # connect to ChromaDB

    print("Knowledge base loaded. Starting evaluation...\n")
    run_evaluation(collection, embedding_model)                     # run the full benchmark