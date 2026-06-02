# predict.py
# Command-line interface for interacting with the Agentic RAG agent
# Lets you type questions and get answers directly in the terminal
# without starting the Flask API
# Analogy: a direct phone line to the agent — no receptionist, no waiting room,
#          just you and the agent having a conversation

import sys                                      # for clean program exit
import os                                       # for environment variables

import config                                   # our central settings file
from knowledge_base import load_knowledge_base  # connects to ChromaDB
from agent import run_agent                     # the full agentic reasoning loop


def print_welcome():
    # Plain explanation: prints a welcome message and lists the available tools
    #                    so the user knows what the agent can do before asking questions
    # Analogy: the opening screen of a video game — here is who you are,
    #          here is what you can do, here is how to quit

    print("\n" + "="*60)
    print("Birdie — Agentic RAG Birding Assistant")
    print("Project 7 | ChromaDB + ReAct Agent")
    print("="*60)
    print("\nI can answer questions about North American backyard birds")
    print("found in Connecticut and Pennsylvania.")
    print("\nI search the knowledge base as many times as needed")
    print("before committing to an answer.")
    print("\nAvailable tools:")
    print("  - search_knowledge_base  (semantic search)")
    print("  - keyword_search         (BM25 keyword search)")
    print("  - filter_by_species      (all chunks about one bird)")
    print("  - compare_species        (two birds side by side)")
    print("\nType 'quit' or 'exit' to stop.")
    print("Type 'verbose' to toggle detailed reasoning output.")
    print("="*60 + "\n")


def print_answer(result, verbose):
    # Plain explanation: prints the agent's final answer and iteration count
    #                    in a clean formatted block
    # Analogy: the verdict at the end of a court case — here is the conclusion
    #          and here is how much deliberation it took to reach it

    print("\n" + "-"*60)
    print("ANSWER:")
    print("-"*60)
    print(result["answer"])                                         # the agent's final answer text
    print("-"*60)
    print(f"Iterations used: {result['iterations']} of {config.MAX_ITERATIONS}")  # how many loops ran
    print("-"*60 + "\n")


def run_cli():
    # Plain explanation: the main interactive loop — loads the knowledge base,
    #                    prints the welcome message, then keeps accepting questions
    #                    until the user types quit or exit
    # Analogy: opening a help desk for the day — set up your tools,
    #          greet the first visitor, and keep helping until closing time

    # Load the knowledge base once at startup
    # This connection stays open for the entire CLI session
    print("Loading knowledge base...")
    collection, embedding_model = load_knowledge_base()             # connect to ChromaDB
    print("Knowledge base loaded successfully.\n")

    # Print the welcome screen
    print_welcome()

    # verbose controls whether the agent prints each iteration as it runs
    # Start with True so the user can see the reasoning loop in action
    verbose = True

    # Keep accepting questions until the user quits
    while True:

        # Prompt the user for input
        # strip() removes any accidental leading or trailing whitespace
        try:
            user_input = input("Your question: ").strip()           # wait for the user to type
        except (KeyboardInterrupt, EOFError):                       # handle Ctrl+C or Ctrl+D gracefully
            print("\n\nGoodbye!")
            sys.exit(0)                                             # exit cleanly with code 0

        # Skip empty input — just re-prompt without doing anything
        if not user_input:
            continue                                                 # go back to the top of the loop

        # Handle the quit command
        if user_input.lower() in ["quit", "exit"]:                  # check for either quit or exit
            print("\nGoodbye!")
            sys.exit(0)                                             # exit cleanly

        # Handle the verbose toggle command
        # Typing 'verbose' flips the verbose flag on or off
        if user_input.lower() == "verbose":
            verbose = not verbose                                   # flip True to False or False to True
            state = "ON" if verbose else "OFF"                      # human-readable state label
            print(f"\nVerbose mode: {state}\n")
            continue                                                 # go back to the top of the loop

        # --- Run the Agent ---
        # The user typed a real question — pass it to the agent
        print(f"\nThinking...\n")

        # run_agent returns a dictionary with the answer, iteration count, and original question
        result = run_agent(
            question=user_input,                                    # the user's question
            collection=collection,                                  # the open ChromaDB collection
            embedding_model=embedding_model,                        # the loaded embedding model
            verbose=verbose,                                        # whether to print each iteration
        )

        # Print the final answer in a clean formatted block
        print_answer(result, verbose)


# --- Entry Point ---
# Runs when you execute `python predict.py` directly
# Skipped when other scripts import from this file
if __name__ == "__main__":
    run_cli()