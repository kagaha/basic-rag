from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from ingestion_pipeline import get_embedding_model  # same BGE model used for ingestion

# Load environment variables (GOOGLE_API_KEY)
load_dotenv()

# Connect to your document database
persistent_directory = "db/chroma_db"
embeddings = get_embedding_model()
db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embeddings,
    collection_metadata={"hnsw:space": "cosine"},
)

# Set up AI model
model = ChatGoogleGenerativeAI(model="gemini-3.6-flash")

# Tune this using similarity_search_with_relevance_scores on on-topic vs off-topic queries
SCORE_THRESHOLD = 0.7

# Store our conversation as messages
chat_history = []


def to_text(response):
    """Gemini can return a list of content blocks; normalise to plain text."""
    content = response.content
    if isinstance(content, list):
        content = "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )
    return content.strip()


def ask_question(user_question):
    print(f"\n--- You asked: {user_question} ---")

    # Step 1: Make the question clear using conversation history
    if chat_history:
        messages = [
            SystemMessage(content="Given the chat history, rewrite the new question to be standalone and searchable. Just return the rewritten question."),
        ] + chat_history + [
            HumanMessage(content=f"New question: {user_question}")
        ]

        result = model.invoke(messages)
        search_question = to_text(result)
        print(f"Searching for: {search_question}")
    else:
        search_question = user_question

    # Step 2: Find relevant documents (with a relevance threshold)
    # retriever = db.as_retriever(
    #     search_type="similarity_score_threshold",
    #     search_kwargs={"k": 3, "score_threshold": SCORE_THRESHOLD},
    # )
    retriever = db.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(search_question)

    # print(f"Found {len(docs)} relevant documents:")
    # for i, doc in enumerate(docs, 1):
    #     lines = doc.page_content.split("\n")[:2]
    #     preview = "\n".join(lines)
    #     print(f"  Doc {i}: {preview}...")

    # No relevant chunks: skip the LLM call
    if not docs:
        answer = "I don't have enough information to answer that question based on the provided documents."
        chat_history.append(HumanMessage(content=user_question))
        chat_history.append(AIMessage(content=answer))
        print(f"Answer: {answer}")
        return answer

    # Step 3: Create final prompt
    context = "\n".join(f"- {doc.page_content}" for doc in docs)

    combined_input = f"""Based on the following documents, please answer this question: {user_question}

Documents:
{context}

Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents, say "I don't have enough information to answer that question based on the provided documents."
"""

    # Step 4: Get the answer
    messages = [
        SystemMessage(content="You are a helpful assistant that answers questions based on provided documents and conversation history."),
    ] + chat_history + [
        HumanMessage(content=combined_input)
    ]

    result = model.invoke(messages)
    answer = to_text(result)

    # Step 5: Remember this conversation
    chat_history.append(HumanMessage(content=user_question))
    chat_history.append(AIMessage(content=answer))

    print(f"Answer: {answer}")
    return answer


# Simple chat loop
def start_chat():
    print("Ask me questions! Type 'quit' to exit.")

    while True:
        question = input("\nYour question: ")

        if question.lower() == "quit":
            print("Goodbye!")
            break

        ask_question(question)


if __name__ == "__main__":
    start_chat()