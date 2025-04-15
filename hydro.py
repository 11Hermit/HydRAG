import os
import json
import fitz  # PyMuPDF
import faiss
import numpy as np
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from tqdm import tqdm
from dotenv import load_dotenv

class RAGSystem:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", chunk_size: int = 500, chunk_overlap: int = 50):
        """Initialize the RAG system with specified parameters."""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = SentenceTransformer(model_name)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.chunks_metadata = {}
        
        # Initialize Gemini
        load_dotenv()
        genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
        self.gemini_model = genai.GenerativeModel('gemini-2.0-flash')

    def process_pdf(self, pdf_path: str) -> List[Dict]:
        """Extract and chunk text from a PDF file."""
        chunks = []
        doc = fitz.open(pdf_path)
        
        text = ""
        for page in doc:
            text += page.get_text()
        
        # Create overlapping chunks
        words = text.split()
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk = " ".join(words[i:i + self.chunk_size])
            chunk_id = f"{os.path.basename(pdf_path)}_{i//self.chunk_size}"
            chunks.append({
                "id": chunk_id,
                "text": chunk,
                "source": pdf_path
            })
        
        return chunks

    def ingest_documents(self, data_dir: str) -> None:
        """Process all PDFs in the data directory and create embeddings."""
        all_chunks = []
        
        # Process all PDFs
        pdf_files = [f for f in os.listdir(data_dir) if f.endswith('.pdf')]
        for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
            pdf_path = os.path.join(data_dir, pdf_file)
            chunks = self.process_pdf(pdf_path)
            all_chunks.extend(chunks)
        
        # Create embeddings and update index
        embeddings = []
        for idx, chunk in enumerate(tqdm(all_chunks, desc="Creating embeddings")):
            embedding = self.embedding_model.encode(chunk["text"])
            embeddings.append(embedding)
            self.chunks_metadata[idx] = chunk
        
        # Add to FAISS index
        embeddings_array = np.array(embeddings).astype('float32')
        self.index.add(embeddings_array)

    def save_index(self, save_dir: str) -> None:
        """Save the FAISS index and metadata to disk."""
        os.makedirs(save_dir, exist_ok=True)
        faiss.write_index(self.index, os.path.join(save_dir, "index.faiss"))
        with open(os.path.join(save_dir, "metadata.json"), 'w') as f:
            json.dump(self.chunks_metadata, f)

    def load_index(self, save_dir: str) -> None:
        """Load the FAISS index and metadata from disk."""
        self.index = faiss.read_index(os.path.join(save_dir, "index.faiss"))
        with open(os.path.join(save_dir, "metadata.json"), 'r') as f:
            self.chunks_metadata = json.load(f)

    def query(self, question: str, k: int = 5) -> Tuple[str, List[Dict]]:
        """Query the RAG system with a question."""
        # Create query embedding
        query_embedding = self.embedding_model.encode(question)
        
        # Search in FAISS
        D, I = self.index.search(query_embedding.reshape(1, -1).astype('float32'), k)
        
        # Get relevant chunks
        relevant_chunks = [self.chunks_metadata[str(i)] for i in I[0]]
        
        # Create context for Gemini
        context = "\n\n".join([chunk["text"] for chunk in relevant_chunks])
        prompt = f"""Based on the following context, please answer the question. 
        If the answer cannot be derived from the context, say so.

        Context:
        {context}

        Question: {question}"""
        
        # Generate response using Gemini
        response = self.gemini_model.generate_content(prompt)
        
        return response.text, relevant_chunks

def main():
    # Example usage
    rag = RAGSystem()
    
    # Create necessary directories
    os.makedirs("data", exist_ok=True)
    os.makedirs("vector_store", exist_ok=True)
    
    # Check if we need to process documents
    if not os.path.exists("vector_store/index.faiss"):
        print("Processing documents...")
        rag.ingest_documents("data")
        rag.save_index("vector_store")
    else:
        print("Loading existing index...")
        rag.load_index("vector_store")
    
    # Interactive query loop
    while True:
        question = input("\nEnter your question (or 'quit' to exit): ")
        if question.lower() == 'quit':
            break
            
        answer, sources = rag.query(question)
        print("\nAnswer:", answer)
        print("\nSources:")
        for source in sources:
            print(f"- {source['source']}")

if __name__ == "__main__":
    main()
