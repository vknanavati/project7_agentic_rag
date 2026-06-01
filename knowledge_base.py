# knowledge_base.py
# Connects to the existing ChromaDB database built in Project 5
# Provides a single function that returns the collection, ready to search
# Analogy: this script is the key that unlocks the library door —
#          it does not build the library, it just opens it

import chromadb  # the vector database client library
from sentence_transformers import SentenceTransformer  # converts text to vectors

import config  # our central settings file


def load_knowledge_base():
    # Plain explanation: connects to the ChromaDB database on disk and returns
    #                    the collection of bird knowledge chunks, plus the
    #                    embedding model needed to search it
    # Analogy: like opening a filing cabinet that someone else already filled —
    #          you didn't do the filing, you're just pulling the drawer open

    # Create a ChromaDB client that reads from the folder on disk
    # PersistentClient means the database is saved to a file, not held in memory
    # If the path is wrong, ChromaDB will create a new empty database there instead
    # of throwing an error — so getting CHROMA_DIR right in config.py matters
    client = chromadb.PersistentClient(path=config.CHROMA_DIR)

    # Open the existing collection by name
    # This must match the collection name Project 5 used when it stored the chunks
    # get_collection() opens an existing one; create_collection() would make a new empty one
    collection = client.get_collection(name=config.CHROMA_COLLECTION_NAME)

    # Load the same embedding model Project 5 used to store the vectors
    # This is critical — if we used a different model to search than was used to store,
    # the vectors would be in different spaces and search results would be meaningless
    # Analogy: if the library catalogued books in French, you need to search in French too
    embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)

    # Return both the collection and the embedding model
    # Every tool in tools.py will need both of these to run searches
    return collection, embedding_model


def get_collection_stats(collection):
    # Plain explanation: returns basic facts about the knowledge base —
    #                    how many chunks are stored and what species are covered
    # Analogy: asking the librarian "how many books do you have, and what topics?"

    # count() returns the total number of chunks stored in the collection
    total_chunks = collection.count()

    # Query ChromaDB for all stored documents to extract the species metadata
    # include=["metadatas"] tells ChromaDB to return the metadata attached to each chunk
    # We stored species names as metadata when building the knowledge base in Project 5
    results = collection.get(include=["metadatas"])

    # Extract the species name from each chunk's metadata
    # Each chunk has a metadata dict; "species" is the key Project 5 used
    # Using a set() automatically removes duplicates — one entry per species
    species = set(
        meta["species"]                    # grab the species name
        for meta in results["metadatas"]   # loop through every chunk's metadata
        if "species" in meta               # skip any chunk that has no species tag
    )

    # Return a summary dictionary with the two key facts
    return {
        "total_chunks": total_chunks,       # total number of searchable chunks
        "num_species": len(species),        # how many distinct bird species are covered
        "species_list": sorted(species),    # alphabetically sorted list of species names
    }


# --- Quick sanity check ---
# When you run this file directly with `python knowledge_base.py`,
# this block executes and prints a summary of the knowledge base
# When other scripts import this file, this block is skipped entirely
if __name__ == "__main__":

    print("Connecting to knowledge base...")  # let the user know we're working

    # Load the collection and embedding model
    collection, embedding_model = load_knowledge_base()

    # Get the stats summary
    stats = get_collection_stats(collection)

    # Print the results
    print(f"Connected successfully.")
    print(f"Total chunks: {stats['total_chunks']}")
    print(f"Species covered: {stats['num_species']}")
    print(f"Species list:")
    for species in stats["species_list"]:       # loop through sorted species names
        print(f"  - {species}")                 # print each one indented