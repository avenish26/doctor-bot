import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_openai import ChatOpenAI  # ✅ Use OpenAI instead of Ollama
import os

# Load OpenAI API key securely from Streamlit Secrets
openai_api_key = st.secrets["OPENAI_API_KEY"]

# Set up Streamlit page
st.set_page_config(page_title="Doctor Bot", page_icon="🩺")
st.title("🧠 AI Doctor Bot (Lite Mode)")
st.markdown("Ask medical questions — answers come from your PDF books.")

# Load FAISS vector index
@st.cache_resource
def load_index():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    return db

db = load_index()

# Set up OpenAI Chat model
llm = ChatOpenAI(api_key=openai_api_key, model="gpt-3.5-turbo", temperature=0)

# Prompt template
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a medical expert. Only use the context below to answer clearly.

Context:
{context}

Question: {question}
Answer:"""
)

# Combine prompt and model
qa_chain = prompt | llm

# User input
question = st.text_input("🩺 Enter a medical question:")

if question:
    with st.spinner("Thinking... please wait..."):
        docs = db.similarity_search(question, k=2)
        context = "\n\n".join([doc.page_content for doc in docs]) if docs else "No matching context found."
        result = qa_chain.invoke({"context": context, "question": question})
        st.markdown("### 🤖 Doctor Bot says:")
        st.write(result)
