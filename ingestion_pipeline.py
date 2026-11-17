import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 100


def normalize_text(text):
    """Use Unix line endings so paragraph breaks ("\\n\\n") are recognised.

    Files saved on Windows use "\\r\\n", which the text splitter would not see
    as a paragraph break.
    """
    return text.replace("\r\n", "\n").replace("\r", "\n")


def documents_from_texts(items):
    """Build Documents from (name, text) pairs held in memory.

    Used by the web app, where text comes from uploads or pasted input rather
    than from a folder on disk. Empty texts are skipped.
    """
    documents = []
    for name, text in items:
        text = normalize_text(text)
        if text.strip():
            documents.append(Document(page_content=text, metadata={"source": name}))
    return documents


def load_documents(docs_path="docs"):
    """Load all text files from the docs directory"""
    print(f"Loading documents from {docs_path}...")

    # Check if docs directory exists
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The directory {docs_path} does not exist. Please create it and add your company files.")

    # Load all .txt files from the docs directory
    loader = DirectoryLoader(
        path=docs_path,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )

    documents = loader.load()

    if len(documents) == 0:
        raise FileNotFoundError(f"No .txt files found in {docs_path}. Please add your company documents.")


    for i, doc in enumerate(documents[:2]):  # Show first 2 documents
        print(f"\nDocument {i+1}:")
        print(f"  Source: {doc.metadata['source']}")
        print(f"  Content length: {len(doc.page_content)} characters")
        print(f"  Content preview: {doc.page_content[:100]}...")
        print(f"  metadata: {doc.metadata}")

    return documents


def looks_like_citations(text):
    """True for chunks that are mostly reference-list entries.

    Wikipedia-style exports end with long lists of citations ("Archived from
    the original on ...", "Retrieved ..."). Their titles often match a question
    better than the real paragraph does, so they crowd out useful chunks.
    """
    return text.count("Retrieved ") >= 2 or text.count("Archived ") >= 2


# Recursive splitting tries paragraph breaks first, then lines, then words, so
# chunks follow the structure of the text. Overlap keeps context across chunk edges.
def split_documents(
    documents,
    chunk_size=DEFAULT_CHUNK_SIZE,
    chunk_overlap=DEFAULT_CHUNK_OVERLAP,
    drop_citation_chunks=True,
    verbose=True,
):
    """Split documents into overlapping chunks, optionally dropping citation lists"""
    if verbose:
        print("Splitting documents into chunks...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = text_splitter.split_documents(documents)

    if drop_citation_chunks:
        kept = [c for c in chunks if not looks_like_citations(c.page_content)]
        if verbose and len(kept) < len(chunks):
            print(f"Dropped {len(chunks) - len(kept)} citation/reference chunks")
        chunks = kept

    if chunks and verbose:

        for i, chunk in enumerate(chunks[:5]):
            print(f"\n--- Chunk {i+1} ---")
            print(f"Source: {chunk.metadata['source']}")
            print(f"Length: {len(chunk.page_content)} characters")
            print(f"Content:")
            print(chunk.page_content)
            print("-" * 50)

        if len(chunks) > 5:
            print(f"\n... and {len(chunks) - 5} more chunks")

    return chunks


def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
            encode_kwargs={"normalize_embeddings": True},  # good for cosine similarity
        )


def create_vector_store(
    chunks,
    persist_directory="db/chroma_db",
    collection_name=None,
    embedding_model=None,
    verbose=True,
):
    """Create a ChromaDB vector store.

    persist_directory=None keeps the store in memory only (used by the web app,
    with one collection per session). Pass embedding_model to reuse an already
    loaded model instead of loading a new one.
    """
    if verbose:
        print("Creating embeddings and storing in ChromaDB...")

    embedding_model = embedding_model or get_embedding_model()

    # Create ChromaDB vector store
    if verbose:
        print("--- Creating vector store ---")
    options = {}
    if collection_name:
        options["collection_name"] = collection_name
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"},
        **options,
    )
    if verbose:
        print("--- Finished creating vector store ---")
        if persist_directory:
            print(f"Vector store created and saved to {persist_directory}")
        else:
            print("Vector store created in memory")
    return vectorstore

def main():
    """Main ingestion pipeline"""
    print("=== RAG Document Ingestion Pipeline ===\n")

    # Define paths
    docs_path = "docs"
    persistent_directory = "db/chroma_db"

    # Check if vector store already exists
    if os.path.exists(persistent_directory):
        print("✅ Vector store already exists. No need to re-process documents.")

        embedding_model = get_embedding_model()
        vectorstore = Chroma(
            persist_directory=persistent_directory,
            embedding_function=embedding_model,
            collection_metadata={"hnsw:space": "cosine"}
        )
        print(f"Loaded existing vector store with {vectorstore._collection.count()} documents")
        return vectorstore

    print("Persistent directory does not exist. Initializing vector store...\n")

    # Step 1: Load documents
    documents = load_documents(docs_path)

    # Step 2: Split into chunks
    chunks = split_documents(documents)

    # # Step 3: Create vector store
    vectorstore = create_vector_store(chunks, persistent_directory)

    print("\n✅ Ingestion complete! Your documents are now ready for RAG queries.")
    return vectorstore

if __name__ == "__main__":
    main()
