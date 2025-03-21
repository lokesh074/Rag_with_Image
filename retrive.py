import os
import chromadb
from langchain.vectorstores import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

# Load ChromaDB (SAME collection used for PDFs & Images)
db_path = "chroma_db"
collection_name = "pdf_chunks"  # SAME collection for PDFs & Images
embed_model = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")
chroma_db = Chroma(collection_name=collection_name, persist_directory=db_path, embedding_function=embed_model)

def retrieve_relevant_text(query, top_k=5):
    """Retrieve top-k relevant text chunks from PDFs & Images stored in ChromaDB."""
    
    # Query ChromaDB for top-k relevant results
    results = chroma_db.similarity_search(query, k=top_k)

    # Display Results
    retrieved_data = []
    print("\n🔎 Retrieved Data:")
    for i, doc in enumerate(results):
        source_type = "📄 PDF" if "image_path" not in doc.metadata else f"🖼️ Image ({doc.metadata['image_path']})"
        
        retrieved_data.append({
            "text": doc.page_content,
            "source": source_type
        })

        print(f"\n📌 Match {i+1}:")
        print("Extracted Text:", doc.page_content[:500])  # Display first 500 chars
        print("Source:", source_type)
        print("-" * 50)

    return retrieved_data  # Return as a list

# Example Query
query = "Explain the role of a data analyst"
retrieved_results = retrieve_relevant_text(query)
