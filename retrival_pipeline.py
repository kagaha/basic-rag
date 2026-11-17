from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI

from ingestion_pipeline import get_embedding_model  # same BGE model used for ingestion

load_dotenv()

persistent_directory = "db/chroma_db"

# Load the same embedding model that built the store
embedding_model = get_embedding_model()

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"},  # a.b / ||a|| * ||b|| = cosine similarity
)

query = "What is weather in Singapore today?"

# 1) Retrieve
# retriever = db.as_retriever(search_kwargs={"k": 5})


retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "k": 5,
        "score_threshold": 0.7 # Only return chunks with cosine similarity ≥ 0.7
    }
)
relevant_docs = retriever.invoke(query)

print(f"User Query: {query}")
# print("--- Context ---")
# for i, doc in enumerate(relevant_docs, 1):
#     print(f"Document {i} ({doc.metadata.get('source')}):\n{doc.page_content}\n")

# 2) Generate the answer with Gemini
llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash")

context = "\n\n".join(doc.page_content for doc in relevant_docs)

prompt = f"""Answer the question using only the context below.
If the answer is not in the context, say "I don't have enough information in the documents."

Context:
{context}

Question: {query}
"""

response = llm.invoke(prompt)

# Thinking models can return a list of content blocks, so normalise to plain text
content = response.content
if isinstance(content, list):
    content = "".join(
        block.get("text", "") if isinstance(block, dict) else str(block)
        for block in content
    )

print("--- Answer ---")
print(content)