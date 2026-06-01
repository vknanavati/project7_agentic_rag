# config.py
# Central configuration file for Project 7: Agentic RAG
# All constants live here so every other script imports from one place
# Analogy: this is the control panel for the entire project — one place to change settings, everywhere benefits

import os  # lets us work with file paths and environment variables

# --- Project Paths ---
# BASE_DIR is the absolute path to the folder this file lives in
# os.path.abspath(__file__) gets the full path to config.py itself
# os.path.dirname() strips the filename and keeps just the folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# The folder where our Flask app will save any output files
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# Path to the ChromaDB database built in Project 5
# We are reusing it directly — no rebuilding needed
# os.path.join builds a file path correctly for any operating system
CHROMA_DIR = os.path.join(
    os.path.dirname(BASE_DIR),          # go up one level from this project folder
    "project5_birding_rag",             # enter the Project 5 folder
    "chroma_db"                         # enter the chroma_db subfolder inside it
)

# Name of the ChromaDB collection that holds our bird knowledge chunks
# This must match exactly what Project 5 used when it created the collection
CHROMA_COLLECTION_NAME = "birding_knowledge"

# --- Embedding Model ---
# The sentence-transformers model used to convert text into vectors
# Must be the same model Project 5 used — different models produce incompatible vectors
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# --- Groq LLM Settings ---
# The LLM we use for the agent's reasoning loop
# llama-3.1-8b-instant is fast and free on Groq's tier — same as Project 5
GROQ_MODEL = "llama-3.1-8b-instant"

# Maximum number of tokens the LLM can generate in a single response
# Keeping this moderate — agent responses are structured, not long essays
GROQ_MAX_TOKENS = 1024

# Temperature controls how creative vs. deterministic the LLM is
# 0.0 = fully deterministic (same input always gives same output)
# We want deterministic reasoning — the agent should think clearly, not creatively
GROQ_TEMPERATURE = 0.0

# --- Retrieval Settings ---
# How many chunks to retrieve from ChromaDB per tool call
# Higher = more context for the LLM, but more tokens consumed
TOP_K_RETRIEVAL = 5

# How many chunks to retrieve when doing a species filter or comparison
# These are more targeted queries so fewer chunks are needed
TOP_K_SPECIES = 4

# Minimum relevance score for a chunk to be included in results
# ChromaDB returns distances (lower = more similar); we convert to scores
# Chunks scoring below this threshold are too loosely related to include
MIN_RELEVANCE_SCORE = 0.3

# --- Agent Settings ---
# Maximum number of reasoning iterations before the agent is forced to stop
# Prevents infinite loops if the LLM never writes "Final Answer"
# Analogy: a timer on a test — you must hand in your paper after 8 attempts whether you're done or not
MAX_ITERATIONS = 8

# The exact string the agent must write to signal it has finished reasoning
# Our loop checks for this string in every LLM response
FINAL_ANSWER_TOKEN = "Final Answer:"

# The exact string the agent writes when it wants to call a tool
# Our parser looks for this to know which tool to run
ACTION_TOKEN = "Action:"

# The exact string the agent writes to provide the tool's input
ACTION_INPUT_TOKEN = "Action Input:"

# The exact string the agent writes to share its reasoning before acting
THOUGHT_TOKEN = "Thought:"

# --- Flask Settings ---
# Port for the Flask API
# Projects have incremented ports to avoid conflicts: 5001, 5002, 5003, 5004
FLASK_PORT = 5004

# Debug mode for Flask — True means Flask will reload on code changes and show detailed errors
# Always set to False in production; True is fine for local development
FLASK_DEBUG = True

# --- Evaluation Settings ---
# Number of test questions to run during evaluation
# Each question tests whether the agent outperforms single-pass RAG
NUM_EVAL_QUESTIONS = 6

# Seconds to wait between evaluation calls to avoid hitting Groq's rate limit
# Groq's free tier allows 6,000 tokens per minute
# Analogy: waiting between orders at a fast-food counter so the kitchen doesn't get overwhelmed
EVAL_SLEEP_SECONDS = 15