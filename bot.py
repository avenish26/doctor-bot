print("✅ Starting offline doctor bot...")

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence

# Load FAISS vector index from your PDF chunks
print("📦 Loading FAISS index...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# Load Mistral LLM from Ollama (fully offline)
llm = OllamaLLM(model="mistral")

# Prompt that accepts 'context' and 'question'
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a professional dermatologist and sexual health expert.

ONLY use the following context (extracted from medical books) to answer the patient's question.

Context:
{context}

Question: {question}
Doctor’s response:"""
)

# Combine prompt + LLM using RunnableSequence
qa_chain = prompt | llm

print("🩺 Doctor Bot (Offline, Mistral) is ready!\n")

# Start Q&A loop
while True:
    question = input("🩺 Your Question: ")
    if question.lower() in ["exit", "quit"]:
        break

    # Get top 4 relevant documents
    docs = db.similarity_search(question, k=4)

    # Flatten documents into a single context string
    context = "\n\n".join([doc.page_content for doc in docs])

    # Pass context + question into chain
    result = qa_chain.invoke({
        "context": context,
        "question": question
    })

    print("\n🤖 Doctor Bot:\n", result, "\n")
