import os
import streamlit as st
import chromadb
from PIL import Image
from langchain.vectorstores import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

# Streamlit UI Setup
st.set_page_config(page_title="RAG PDF & Image Retrieval", layout="wide")
st.title("🔎 RAG-Based PDF & Image Retrieval")

# Load ChromaDB (Unified for PDFs & Images)
db_path = "chroma_db"
collection_name = "pdf_chunks"
embed_model = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")
chroma_db = Chroma(collection_name=collection_name, persist_directory=db_path, embedding_function=embed_model)

def retrieve_relevant_text(query, top_k=5):
    """Retrieve top-k relevant text chunks from PDFs & Images stored in ChromaDB."""
    results = chroma_db.similarity_search(query, k=top_k)
    retrieved_data = []

    for doc in results:
        source_type = "📄 PDF"
        image_path = None
        if "image_path" in doc.metadata:
            source_type = f"🖼️ Image"
            image_path = doc.metadata["image_path"]

        retrieved_data.append({
            "text": doc.page_content,
            "source": source_type,
            "image_path": image_path
        })

    return retrieved_data

# Streamlit Input
query = st.text_input("🔍 Enter your query:", "")

if query:
    st.subheader("📌 Retrieved Results")
    
    retrieved_results = retrieve_relevant_text(query)
    
    for idx, result in enumerate(retrieved_results):
        st.markdown(f"### ✅ Match {idx+1}")
        st.write(f"**Source:** {result['source']}")
        st.write(f"**Extracted Text:** {result['text'][:500]}...")  # Display first 500 chars

        # If source is an image, display it
        if result["image_path"]:
            image_path = result["image_path"]
            if os.path.exists(image_path):
                image = Image.open(image_path)
                st.image(image, caption=f"🖼️ Retrieved Image: {image_path}", use_column_width=True)
            else:
                st.error(f"⚠️ Image not found: {image_path}")

    if not retrieved_results:
        st.warning("⚠️ No relevant results found.")
