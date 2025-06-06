import os
import uuid
import pymupdf
from pathlib import Path
from dotenv import load_dotenv
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Qdrant Setup
qdrant = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),    
)

COLLECTION_NAME = "learning_vectors"

# Step 1: Load PDF and extract text
def load_pdf_as_documents(file_path):
    
    loader = PyPDFLoader(file_path= file_path)
    docs = loader.load() # Read PDF file

    return docs

# Step 2: Chunking
def split_documents(documents, chunk_size=1000, chunk_overlap=400):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)

# Step 3: Vector Embedding and Qdrant Upload
def index_documents(documents, collection_name):
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    vector_store = QdrantVectorStore.from_documents(
        documents=documents,
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
        collection_name=collection_name,
        embedding=embedding_model
    )
    
    print(f"[INFO] Indexed {len(documents)} documents into Qdrant.")
    return vector_store

# Step 4: Search Function
def search_query(query, collection_name):
    embedder = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    query_vec = embedder.embed_query(query)
    results = qdrant.query_points(
        collection_name=collection_name,
        query=query_vec,
        limit=15,
        with_payload=True
    )
    return results.points

# Step 5: Create Prompt Context
def build_prompt_context(results):
    context = "\n\n\n".join([
    f"Page Content: {result.payload['page_content']}\n"
    f"Page Number: {result.payload['metadata']['page_label']}\n"
    f"File Location: {result.payload['metadata']['source']}\n"
    f"Score: {result.score}\n"
    f"Total Pages: {result.payload['metadata']['total_pages']}"
    for result in results
    ])
    return f"""
    You are a helpfull AI Assistant who asnweres user query based on the available context
    retrieved from a PDF file along with page_contents and page number.

    You should only ans the user based on the following context and navigate the user
    to open the right page number to know more.

    Context:
    {context}
"""

# Step 6: Interactive Chat

def chat_with_bot(system_prompt,query):
    chat_history = []
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        system_instruction=system_prompt
    )

    chat = model.start_chat(history=chat_history)
    response = chat.send_message(query)
    chat_history.append({"role": "user", "parts": [query]})
    chat_history.append({"role": "model", "parts": [response.text]})
    return response

def chat_loop():
    print("📘 PDF Assistant is ready! Type your query or 'exit' to quit.")
 
    
    while True:
        query = input("👨 > ")
        if query.lower() in ["exit", "quit", "q"]:
            print("👋 Bye! Have a great day.")
            break
        
        results = search_query(query, COLLECTION_NAME)
        system_prompt = build_prompt_context(results)        
        response = chat_with_bot(system_prompt,query)        
        print(f"🤖 : {response.text}")
        

# Main Flow
if __name__ == "__main__":
    file_path = Path(__file__).parent / "nodejs.pdf"
    print("📥 Reading and indexing PDF...")
    docs = load_pdf_as_documents(file_path)
    chunks = split_documents(docs)
    index_documents(chunks, COLLECTION_NAME)
    chat_loop()