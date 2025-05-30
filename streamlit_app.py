import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence

# Page setup
st.set_page_config(page_title="Doctor Bot", page_icon="🩺")
st.title("🧠 AI Doctor Bot (Lite Mode)")
st.markdown("Ask medical questions — answers come from your PDF books.")

# Load FAISS vector index (cached)
@st.cache_resource
def load_index():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    return db

db = load_index()

# Use lighter model (Mistral or Tiny model if needed)
llm = OllamaLLM(model="mistral")

# Optimized prompt
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a medical expert. Only use the context below to answer clearly.

Context:
{context}

Question: {question}
Answer:"""
)

qa_chain = prompt | llm

# Input box
question = st.text_input("🩺 Enter a medical question:")

if question:
    with st.spinner("Thinking... please wait..."):
        docs = db.similarity_search(question, k=1)  # 🔽 reduced from k=4
        context = docs[0].page_content if docs else "No matching info found."
        result = qa_chain.invoke({"context": context, "question": question})
        st.markdown("### 🤖 Doctor Bot says:")
        st.write(result)
