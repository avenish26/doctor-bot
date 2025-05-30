from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from PyPDF2 import PdfReader
import os

def load_pdf_text(path):
    text = ""
    reader = PdfReader(path)
    for i, page in enumerate(reader.pages):
        if i >= 100:  # 🟡 optional: limit to 100 pages for speed
            break
        content = page.extract_text()
        if content:
            text += content + "\n"
    return text

# Load all PDFs
all_texts = []
for filename in os.listdir("docs"):
    if filename.endswith(".pdf"):
        path = os.path.join("docs", filename)
        print(f"📖 Reading: {filename}")
        all_texts.append(load_pdf_text(path))

# Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=300)
docs = []
for text in all_texts:
    print(f"📦 Splitting and embedding: {len(text)} characters")
    docs.extend(splitter.create_documents([text]))

# ✅ Final fix — pass model_name and use correct import
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.from_documents(docs, embeddings)
db.save_local("faiss_index")

print("✅ Done! All PDFs have been indexed.")
