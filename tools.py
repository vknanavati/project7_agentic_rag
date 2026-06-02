# tools.py
# Defines the four retrieval tools the agent can call during its reasoning loop
# Each tool searches the bird knowledge base in a different way
# Analogy: these are the different search strategies a librarian can use —
#          browse by topic, search by keyword, pull everything on one author,
#          or pull two authors side by side for comparison

import numpy as np                              # for numerical operations on scores
from rank_bm25 import BM25Okapi                 # keyword-based search algorithm
from sentence_transformers import SentenceTransformer  # for converting text to vectors

import config                                   # our central settings file


# --- Tool Schemas ---
# These descriptions are injected into the system prompt so the LLM
# knows what tools exist, what each one does, and when to use it
# These are plain-English descriptions of each tool, written for the LLM to read
# The agent reads these at the start of every reasoning loop and uses them
# to decide which tool fits the current question
# Analogy: a job posting — the LLM reads what each tool does and picks the right one
# for the task, the same way a manager picks the right employee for a job

TOOL_SCHEMAS = {
    "search_knowledge_base": {
        "name": "search_knowledge_base",
        "description": (
            "Performs semantic search over the bird knowledge base. "
            "Best for open-ended or conceptual questions where the exact wording "
            "of the query may differ from the stored text. "
            "Use this as your default search tool."
        ),
        "argument": "A natural language search query describing what you are looking for.",
        "example": "search_knowledge_base('migration patterns of warblers')",
    },
    "keyword_search": {
        "name": "keyword_search",
        "description": (
            "Performs BM25 keyword search over the bird knowledge base. "
            "Best when the question contains specific technical terms, species names, "
            "or exact phrases that are likely to appear verbatim in the documents. "
            "Use this when semantic search returns weak or off-topic results."
        ),
        "argument": "A short keyword query — specific terms or phrases to match exactly.",
        "example": "keyword_search('suet feeder winter calories')",
    },
    "filter_by_species": {
        "name": "filter_by_species",
        "description": (
            "Retrieves all knowledge base chunks about a single named bird species. "
            "Best for deep, species-specific questions where you want everything "
            "the knowledge base knows about one bird. "
            "Use this when the question is clearly about one specific species."
        ),
        "argument": "The exact species name using underscores and lowercase (e.g. 'black_capped_chickadee').",
        "example": "filter_by_species('black_capped_chickadee')",
    },
    "compare_species": {
        "name": "compare_species",
        "description": (
            "Retrieves chunks about two named bird species side by side. "
            "Best for comparison questions asking how two birds differ or are similar. "
            "Use this when the question explicitly compares two species."
        ),
        "argument": "Two species names in lowercase with underscores, separated by a comma (e.g. 'blue_jay, american_crow').",
        "example": "compare_species('blue_jay, american_crow')",
    },
}


def format_tool_schemas_for_prompt():
    # Plain explanation: converts the TOOL_SCHEMAS dictionary into a single
    #                    formatted string that can be injected into the system prompt
    # Analogy: printing the job postings onto a single sheet of paper
    #          so the manager (LLM) can read all the options at once

    lines = []                                  # will hold each line of the formatted output
    for tool in TOOL_SCHEMAS.values():          # loop through each tool definition
        lines.append(f"Tool: {tool['name']}")               # tool name
        lines.append(f"Description: {tool['description']}") # what it does
        lines.append(f"Argument: {tool['argument']}")        # what input it expects
        lines.append(f"Example: {tool['example']}")          # a usage example
        lines.append("")                                      # blank line between tools

    return "\n".join(lines)                     # join all lines into one string


# --- Tool 1: Semantic Search ---

def search_knowledge_base(query, collection, embedding_model):
    # Plain explanation: converts the query into a vector and finds the chunks
    #                    in ChromaDB whose vectors are most similar to it
    # Analogy: describing what you are looking for in your own words,
    #          and the librarian finds the books that best match that description
    #          even if they use different wording

    # Convert the query string into a vector using the embedding model
    # tolist() converts the numpy array to a plain Python list that ChromaDB expects
    query_vector = embedding_model.encode(query).tolist()

    # Search ChromaDB for the top K most similar chunks
    # query_embeddings takes a list of vectors; we wrap ours in a list since we have one query
    # n_results is how many chunks to return
    # include tells ChromaDB what data to return alongside the chunk IDs
    results = collection.query(
        query_embeddings=[query_vector],        # our query vector
        n_results=config.TOP_K_RETRIEVAL,       # how many chunks to return
        include=["documents", "metadatas", "distances"],  # return text, metadata, and similarity scores
    )

    # Format the results into a clean list of dictionaries
    return _format_results(results)


# --- Tool 2: Keyword Search ---

def keyword_search(query, collection, embedding_model):
    # Plain explanation: retrieves all chunks from ChromaDB, then uses BM25
    #                    to rank them by how well they match the query keywords
    # Analogy: scanning every page of every book in the library for exact words,
    #          then ranking the pages by how many matching words they contain

    # Fetch all documents and their metadata from ChromaDB
    # We need all chunks so BM25 can rank them
    all_data = collection.get(include=["documents", "metadatas"])

    all_docs = all_data["documents"]            # list of all chunk text strings
    all_metas = all_data["metadatas"]           # list of all chunk metadata dicts

    # Tokenize each document into a list of lowercase words
    # BM25 works on word lists, not raw strings
    tokenized_docs = [doc.lower().split() for doc in all_docs]

    # Build the BM25 index from the tokenized documents
    # This index knows word frequencies across all documents
    bm25 = BM25Okapi(tokenized_docs)

    # Tokenize the query the same way
    tokenized_query = query.lower().split()

    # Score every document against the query
    # Higher score = better keyword match
    scores = bm25.get_scores(tokenized_query)

    # Get the indices of the top K highest-scoring documents
    # np.argsort sorts ascending, so we reverse with [::-1] to get descending
    top_indices = np.argsort(scores)[::-1][:config.TOP_K_RETRIEVAL]

    # Build a results list in the same format as search_knowledge_base
    # so the agent receives consistent output regardless of which tool it called
    formatted = []
    for idx in top_indices:                     # loop through the top K indices
        formatted.append({
            "text": all_docs[idx],              # the chunk text
            "metadata": all_metas[idx],         # species name and other metadata
            "score": float(scores[idx]),        # the BM25 relevance score
        })

    return formatted


# --- Tool 3: Filter by Species ---

def filter_by_species(species_name, collection, embedding_model):
    # Plain explanation: retrieves all chunks that are tagged with a specific
    #                    species name in their metadata — ignores all other chunks
    # Analogy: going to the library and asking for every book by one specific author,
    #          regardless of what topic each book covers

    # Query ChromaDB using a metadata filter
    # where={"species": species_name} tells ChromaDB to only return chunks
    # whose metadata has that exact species value
    # n_results caps how many we return even if more exist
    results = collection.query(
        query_embeddings=[                          # ChromaDB requires a query vector even for metadata filters
            embedding_model.encode(species_name).tolist()  # encode the species name as a dummy vector
        ],
        n_results=config.TOP_K_SPECIES,             # return up to TOP_K_SPECIES chunks
        where={"species": species_name},            # only return chunks tagged with this species
        include=["documents", "metadatas", "distances"],
    )

    return _format_results(results)


# --- Tool 4: Compare Species ---

def compare_species(species_input, collection, embedding_model):
    # Plain explanation: splits the input into two species names, retrieves chunks
    #                    for each one separately, and returns them combined
    # Analogy: asking the librarian for books by Author A and Author B
    #          so you can read them side by side and compare

    # Split the input string on the comma to get the two species names
    # strip() removes any accidental whitespace around the names
    parts = [s.strip() for s in species_input.split(",")]

    # If the agent did not provide exactly two species, return an error message
    # The agent will read this and know it needs to try again with the right format
    if len(parts) != 2:
        return [{"text": "Error: compare_species requires exactly two species names separated by a comma.", "metadata": {}, "score": 0.0}]

    species_a, species_b = parts[0], parts[1]   # unpack the two species names

    # Retrieve chunks for each species using the metadata filter
    results_a = collection.query(
        query_embeddings=[embedding_model.encode(species_a).tolist()],
        n_results=config.TOP_K_SPECIES,
        where={"species": species_a},
        include=["documents", "metadatas", "distances"],
    )

    results_b = collection.query(
        query_embeddings=[embedding_model.encode(species_b).tolist()],
        n_results=config.TOP_K_SPECIES,
        where={"species": species_b},
        include=["documents", "metadatas", "distances"],
    )

    # Format both result sets and combine them into one list
    # The agent receives all chunks for both species in a single tool response
    return _format_results(results_a) + _format_results(results_b)


# --- Tool Registry ---
# Maps tool name strings to the actual Python functions
# The agent writes a tool name as text; this dictionary converts that text
# into a callable function
# Analogy: a phone directory — the agent looks up the name and gets the number to call
TOOL_REGISTRY = {
    "search_knowledge_base": search_knowledge_base,
    "keyword_search": keyword_search,
    "filter_by_species": filter_by_species,
    "compare_species": compare_species,
}


def call_tool(tool_name, tool_input, collection, embedding_model):
    # Plain explanation: looks up the tool name in TOOL_REGISTRY and calls
    #                    the matching function with the agent's input
    # Analogy: a switchboard operator — receives the name of who to connect to,
    #          finds their line, and puts the call through

    # Check if the tool name the agent wrote actually exists in our registry
    if tool_name not in TOOL_REGISTRY:
        # Return an error message the agent can read and recover from
        return [{"text": f"Error: '{tool_name}' is not a valid tool. Choose from: {list(TOOL_REGISTRY.keys())}", "metadata": {}, "score": 0.0}]

    # Look up and call the function, passing the input and knowledge base objects
    tool_fn = TOOL_REGISTRY[tool_name]          # get the function object
    return tool_fn(tool_input, collection, embedding_model)  # call it and return results


# --- Helper: Format ChromaDB Results ---

def _format_results(results):
    # Plain explanation: converts ChromaDB's nested result structure into a clean
    #                    flat list of dictionaries that every tool returns consistently
    # Analogy: ChromaDB hands back a messy cardboard box of papers;
    #          this function sorts them into a neat folder with labeled tabs

    # ChromaDB returns nested lists because it supports batch queries
    # results["documents"][0] is the list of documents for our single query
    documents = results["documents"][0]         # list of chunk text strings
    metadatas = results["metadatas"][0]         # list of metadata dicts
    distances = results["distances"][0]         # list of distance scores (lower = more similar)

    formatted = []                              # will hold our clean result dictionaries
    for doc, meta, dist in zip(documents, metadatas, distances):  # loop through all three lists together

        # Convert ChromaDB distance to a relevance score between 0 and 1
        # Distance is how far apart two vectors are — lower distance = higher similarity
        # We convert: score = 1 / (1 + distance) so that distance 0 = score 1.0
        score = 1 / (1 + dist)

        # Only include chunks that meet our minimum relevance threshold
        if score >= config.MIN_RELEVANCE_SCORE:
            formatted.append({
                "text": doc,                    # the raw chunk text
                "metadata": meta,               # species name and other stored metadata
                "score": round(score, 4),       # relevance score rounded to 4 decimal places
            })

    return formatted                            # return the clean list