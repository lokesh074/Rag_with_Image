import chromadb
from langchain.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain.vectorstores import Chroma

# 1️⃣ Load PDF
pdf_path = "data_analyst.pdf"
loader = PyPDFLoader(pdf_path)
documents = loader.load()

# 2️⃣ Load Embedding Model
embed_model = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")

# 3️⃣ Chunk the Text using SemanticChunker
semantic_chunker = SemanticChunker(embed_model, breakpoint_threshold_type="percentile")
semantic_chunks = semantic_chunker.create_documents([d.page_content for d in documents])

# 4️⃣ Initialize ChromaDB
db_path = "chroma_db"
chroma_db = Chroma(collection_name="pdf_chunks", persist_directory=db_path, embedding_function=embed_model)

# 5️⃣ Store Chunks in ChromaDB
chroma_db.add_documents(semantic_chunks)

# Save the database locally
chroma_db.persist()
print(f"✅ {len(semantic_chunks)} chunks saved to ChromaDB at '{db_path}'")
