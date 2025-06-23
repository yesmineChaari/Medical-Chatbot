
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore as Qdrant
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
import textwrap

# Configuration
QDRANT_COLLECTION_NAME = "medical_qa"
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-base"

# 1. Connect to Qdrant
qdrant_client = QdrantClient(host="localhost", port=6333)

# 2. Set up vectorstore using LangChain wrapper
embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

vectorstore = Qdrant(
    client=qdrant_client,
    collection_name=QDRANT_COLLECTION_NAME,
    embedding=embedding_model,
    content_payload_key="question"
)

# 3. Load LLaMA 3 using Ollama
llm = OllamaLLM(model="llama3", temperature=0.3)

# 4. Custom Prompt Template
rag_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=textwrap.dedent("""
You are a helpful, careful, and empathetic AI medical assistant.

You must answer **only based on the context provided below**, and **only if it directly answers the question**.
Do **not** assume, guess, summarize, or generalize from unrelated information.

If the question involves medication, clearly state that the user should consult a doctor or pharmacist.
Do **not** provide exact dosages, treatment plans, or medical diagnoses.

If the context is **unclear**, **irrelevant**, or **incomplete**, respond with:

"I recommend speaking to a medical professional for accurate advice."

---

QUESTION:
{question}

CONTEXT SNIPPETS (from reliable medical sources):
{context}

---

INSTRUCTIONS TO THE MODEL:
- Use only the parts of the context that are directly relevant to the question.
- Do not include unrelated or partially relevant details.
- Do not combine or assume relationships between multiple snippets unless explicitly stated.
- Do not repeat context unnecessarily.
- Do not explain why the context is missing or inadequate. Do not mention lack of information.
- The response should be concise, clear, and directly answer the question based on the provided context.
- Output ONLY the FINAL ANSWER section. Do not restate the question or summarize the reasoning.
FINAL ANSWER:
""")
)

# 5. Create the retriever
retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 5, "score_threshold": 0.75}
)

# 6. Custom RAG with prompt
def generate_safe_answer(user_question: str):
    # Retrieve relevant documents
    docs = retriever.invoke(user_question)


    if not docs:
        return "I'm sorry, I couldn't find relevant information. Please consult a medical professional."

    # Format the context snippets
    context_snippets = "\n\n".join([f"- {doc.page_content}" for doc in docs])

    # Build final prompt
    prompt = rag_prompt.format(context=context_snippets, question=user_question)

    # Call LLM
    response = llm.invoke(prompt)

    return response

# 7. Chat loop
print(" Medical Chatbot is ready! Type 'exit' to quit.\n")
while True:
    user_question = input("You: ")
    if user_question.lower() in ["exit", "quit"]:
        break

    answer = generate_safe_answer(user_question)
    print(f"\nBot: {answer}\n")
