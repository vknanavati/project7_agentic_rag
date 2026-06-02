# app.py
# Flask API for the Agentic RAG birding assistant
# Exposes the agent as HTTP endpoints so any client can send questions
# and receive answers without using the command line
# Analogy: the front desk of a hotel — clients walk up, make a request,
#          and the front desk handles routing it to the right department

import time                                     # for tracking response time
from flask import Flask, request, jsonify       # Flask web framework and request/response tools

import config                                   # our central settings file
from knowledge_base import (                    # knowledge base connection functions
    load_knowledge_base,
    get_collection_stats,
)
from agent import run_agent                     # the full agentic reasoning loop


# --- App Initialization ---
# Create the Flask application instance
# __name__ tells Flask where to look for templates and static files
app = Flask(__name__)

# --- Load Knowledge Base at Startup ---
# Load once when the server starts — stays in memory for the entire session
# Every incoming request reuses this open connection
# Analogy: the hotel front desk setting up their computer at the start of the shift —
#          done once, used all day
print("Loading knowledge base...")
collection, embedding_model = load_knowledge_base()     # connect to ChromaDB
print("Knowledge base loaded. Server ready.\n")


# --- Routes ---

@app.route("/health", methods=["GET"])
def health():
    # Plain explanation: a simple endpoint that confirms the server is running
    #                    and the knowledge base is connected
    # Analogy: calling the hotel to ask "are you open?" before making a reservation

    # Get basic stats about the knowledge base to include in the health response
    stats = get_collection_stats(collection)            # total chunks, species count, species list

    return jsonify({
        "status": "ok",                                 # server is running
        "model": config.GROQ_MODEL,                     # which LLM is being used
        "embedding_model": config.EMBEDDING_MODEL,      # which embedding model is loaded
        "knowledge_base": {
            "total_chunks": stats["total_chunks"],      # how many chunks are stored
            "num_species": stats["num_species"],        # how many bird species are covered
        },
        "max_iterations": config.MAX_ITERATIONS,        # agent iteration limit
        "flask_port": config.FLASK_PORT,                # which port the server is on
    })


@app.route("/ask", methods=["POST"])
def ask():
    # Plain explanation: receives a question as JSON, runs it through the agent,
    #                    and returns the answer along with metadata about how
    #                    many iterations the agent used
    # Analogy: a guest walks up to the front desk with a question —
    #          the front desk takes the question, finds the answer,
    #          and reports back with the answer and how long it took

    # Parse the incoming JSON request body
    data = request.get_json()                           # read the JSON body of the POST request

    # Validate that a question was provided
    if not data or "question" not in data:
        return jsonify({
            "error": "Request body must include a 'question' field."
        }), 400                                         # 400 = Bad Request

    # Extract the question from the request body
    question = data["question"].strip()                 # remove leading/trailing whitespace

    # Reject empty questions
    if not question:
        return jsonify({
            "error": "Question cannot be empty."
        }), 400                                         # 400 = Bad Request

    # Extract optional verbose flag — defaults to False for API responses
    # When True, the agent prints each iteration to the server console
    verbose = data.get("verbose", False)                # default to False if not provided

    # Record the start time so we can report how long the agent took
    start_time = time.time()                            # current time in seconds

    # --- Run the Agent ---
    result = run_agent(
        question=question,                              # the user's question
        collection=collection,                         # the open ChromaDB collection
        embedding_model=embedding_model,               # the loaded embedding model
        verbose=verbose,                               # whether to print iterations to console
    )

    # Calculate how long the agent took to answer
    elapsed = round(time.time() - start_time, 2)       # seconds, rounded to 2 decimal places

    # Return the answer and metadata as JSON
    return jsonify({
        "question": question,                           # echo the original question
        "answer": result["answer"],                     # the agent's final answer
        "iterations": result["iterations"],             # how many reasoning loops ran
        "max_iterations": config.MAX_ITERATIONS,        # the cap we set in config
        "response_time_seconds": elapsed,               # how long it took end to end
    })


@app.route("/species", methods=["GET"])
def species():
    # Plain explanation: returns the list of all bird species covered by the knowledge base
    #                    so clients know what topics the agent can answer questions about
    # Analogy: the hotel handing you a list of all their services
    #          so you know what to ask for before approaching the front desk

    # Get the full stats including the species list
    stats = get_collection_stats(collection)

    return jsonify({
        "num_species": stats["num_species"],            # total number of species
        "species": stats["species_list"],               # alphabetically sorted list of species names
    })


@app.route("/compare", methods=["POST"])
def compare():
    # Plain explanation: receives two species names and runs the agent with a
    #                    pre-formed comparison question so clients don't have to
    #                    phrase comparison questions themselves
    # Analogy: a shortcut button at the front desk — instead of explaining
    #          what you want from scratch, you press the compare button
    #          and the desk knows exactly what to do

    # Parse the incoming JSON request body
    data = request.get_json()

    # Validate that both species fields were provided
    if not data or "species_a" not in data or "species_b" not in data:
        return jsonify({
            "error": "Request body must include 'species_a' and 'species_b' fields."
        }), 400                                         # 400 = Bad Request

    # Extract the two species names
    species_a = data["species_a"].strip()               # first species name
    species_b = data["species_b"].strip()               # second species name

    # Build a natural language comparison question from the two species names
    # This gets passed directly to the agent as if the user typed it
    question = (
        f"Compare {species_a} and {species_b}. "
        f"How are they similar and how are they different in terms of "
        f"behavior, diet, habitat, and appearance?"
    )

    # Record start time
    start_time = time.time()

    # Run the agent with the constructed comparison question
    result = run_agent(
        question=question,
        collection=collection,
        embedding_model=embedding_model,
        verbose=False,                                  # keep console clean for API use
    )

    elapsed = round(time.time() - start_time, 2)       # total response time

    return jsonify({
        "species_a": species_a,                         # echo first species
        "species_b": species_b,                         # echo second species
        "question": question,                           # the constructed question
        "answer": result["answer"],                     # the agent's comparison answer
        "iterations": result["iterations"],             # how many loops ran
        "response_time_seconds": elapsed,               # how long it took
    })


# --- Entry Point ---
# Runs when you execute `python app.py` directly
if __name__ == "__main__":
    app.run(
        host="0.0.0.0",                                 # accept connections from any network interface
        port=config.FLASK_PORT,                         # port 5004 as set in config
        debug=config.FLASK_DEBUG,                       # True for local development
    )