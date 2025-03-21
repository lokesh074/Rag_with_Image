import os
import google.generativeai as genai
import chromadb
from langchain.vectorstores import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

# Configure Google Gemini API
genai.configure(api_key="AIzaSyBIhPtiJ412v2U3LqHwWRttA6Jv6tQrS54")

# Use the same collection for PDF and image text
db_path = "chroma_db"
collection_name = "pdf_chunks"  # SAME collection as PDF chunks
embed_model = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")
chroma_db = Chroma(collection_name=collection_name, persist_directory=db_path, embedding_function=embed_model)

def extract_text_from_image(image_path):
    """Extracts text from an image using Google Gemini AI's OCR."""
    
    # Upload image to Gemini API
    sample_file = genai.upload_file(path=image_path)

    # OCR request
    text = "Explain given image in detail."
    model = genai.GenerativeModel(model_name="models/gemini-1.5-flash-002")
    response = model.generate_content([text, sample_file])

    return response.text  # Return extracted text

def process_images_and_store():
    """Processes all images in the 'images' directory and stores extracted text in ChromaDB (same collection as PDFs)."""
    image_dir = "images"
    if not os.path.exists(image_dir):
        print("❌ Images directory not found!")
        return

    for filename in os.listdir(image_dir):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(image_dir, filename)
            print(f"🔍 Processing {image_path}...")

            # Extract text
            extracted_text = extract_text_from_image(image_path)

            # Store in the same collection as PDFs
            chroma_db.add_texts(
                texts=[extracted_text],
                metadatas=[{"source": "image", "image_path": image_path}]
            )

    # Save database
    chroma_db.persist()
    print("✅ All extracted texts from images saved in 'pdf_chunks' collection!")

# Run the function
process_images_and_store()
