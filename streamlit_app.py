import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI  # ✅ Cloud-friendly model
from langchain_core.runnables import RunnableSequence

# Securely access your API key from Streamlit secrets
openai_api_key = st.secrets["OPENAI_API_KEY"]

# Page setup
st.set_page_config(page_title="Doctor Bot", page_icon="🩺")
st.title("🧠 AI Doctor Bot (Lite Mode)")
st.markdown("Ask medical questions — answers come from your PDF books.")

# Load vector index
@st.cache_resource
def load_index():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

db = load_index()

# Load OpenAI LLM (cloud-safe)
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key=openai_api_key)

# Prompt
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a medical expert. Use the context below to answer clearly.

Context:
{context}

Question: {question}
Answer:"""
)

qa_chain = prompt | llm  # Chain using RunnableSequence

# UI
question = st.text_input("🩺 Enter a medical question:")

if question:
    with st.spinner("Thinking..."):
        docs = db.similarity_search(question, k=2)
        context = "\n\n".join([doc.page_content for doc in docs]) if docs else "No match found."
        result = qa_chain.invoke({"context": context, "question": question})
        st.markdown("### 🤖 Doctor Bot says:")
        st.write(result)
