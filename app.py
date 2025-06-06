import os
import streamlit as st
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
import tempfile
import time

# Page configuration
st.set_page_config(
    page_title="PDF Chat Assistant",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load environment variables
load_dotenv()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "collection_name" not in st.session_state:
    st.session_state.collection_name = "learning_vectors"

# Configure Google AI
@st.cache_resource
def configure_google_ai():
    try:
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        return True
    except Exception as e:
        st.error(f"Error configuring Google AI: {e}")
        return False

# Qdrant Setup with timeout configuration
@st.cache_resource
def setup_qdrant():
    try:
        qdrant = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
            timeout=120  # Increase timeout to 2 minutes
        )
        return qdrant
    except Exception as e:
        st.error(f"Error connecting to Qdrant: {e}")
        return None

# Clear existing collection
def clear_collection(collection_name, qdrant_client):
    """Clear/delete existing collection if it exists"""
    try:
        # Check if collection exists
        collections = qdrant_client.get_collections()
        existing_collections = [col.name for col in collections.collections]
        
        if collection_name in existing_collections:
            qdrant_client.delete_collection(collection_name)
            st.info(f"🗑️ Cleared existing collection: {collection_name}")
            time.sleep(2)  # Wait for deletion to complete
        else:
            st.info(f"📝 Creating new collection: {collection_name}")
    except Exception as e:
        st.warning(f"Warning clearing collection: {e}")

# Load PDF and extract text
def load_pdf_as_documents(file_path):
    try:
        loader = PyPDFLoader(file_path=file_path)
        docs = loader.load()
        return docs
    except Exception as e:
        st.error(f"Error loading PDF: {e}")
        return None

# Chunking
def split_documents(documents, chunk_size=1000, chunk_overlap=400):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(documents)

# Vector Embedding and Qdrant Upload with retry mechanism
def index_documents_with_retry(documents, collection_name, qdrant_client, max_retries=3):
    """Index documents with retry mechanism and batch processing"""
    
    # Clear existing collection first
    clear_collection(collection_name, qdrant_client)
    
    for attempt in range(max_retries):
        try:
            st.info(f"🔄 Indexing attempt {attempt + 1}/{max_retries}")
            
            # Process in smaller batches to avoid timeout
            batch_size = min(50, len(documents))  # Process max 50 documents at a time
            total_batches = (len(documents) + batch_size - 1) // batch_size
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Initialize embedding model
            embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            
            for batch_idx in range(total_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(documents))
                batch_docs = documents[start_idx:end_idx]
                
                status_text.text(f"Processing batch {batch_idx + 1}/{total_batches} ({len(batch_docs)} documents)")
                
                # Create vector store for this batch
                if batch_idx == 0:
                    # First batch creates the collection
                    vector_store = QdrantVectorStore.from_documents(
                        documents=batch_docs,
                        url=os.getenv("QDRANT_URL"),
                        api_key=os.getenv("QDRANT_API_KEY"),
                        collection_name=collection_name,
                        embedding=embedding_model,
                        timeout=120
                    )
                else:
                    # Subsequent batches add to existing collection
                    temp_store = QdrantVectorStore.from_documents(
                        documents=batch_docs,
                        url=os.getenv("QDRANT_URL"),
                        api_key=os.getenv("QDRANT_API_KEY"),
                        collection_name=collection_name,
                        embedding=embedding_model,
                        timeout=120
                    )
                
                # Update progress
                progress = (batch_idx + 1) / total_batches
                progress_bar.progress(progress)
                
                # Small delay between batches
                time.sleep(1)
            
            progress_bar.empty()
            status_text.empty()
            
            st.success(f"✅ Successfully indexed {len(documents)} documents!")
            return vector_store if 'vector_store' in locals() else True
            
        except Exception as e:
            error_msg = f"Attempt {attempt + 1} failed: {str(e)}"
            st.warning(error_msg)
            
            if attempt < max_retries - 1:
                st.info(f"⏳ Retrying in 5 seconds...")
                time.sleep(5)
            else:
                st.error(f"❌ Failed to index documents after {max_retries} attempts")
                return None

# Search Function
def search_query(query, collection_name, qdrant_client):
    try:
        embedder = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        query_vec = embedder.embed_query(query)
        results = qdrant_client.query_points(
            collection_name=collection_name,
            query=query_vec,
            limit=15,
            with_payload=True,
            timeout=30
        )
        return results.points
    except Exception as e:
        st.error(f"Error searching: {e}")
        return []

# Create Prompt Context
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
    You are a helpful AI Assistant who answers user queries based on the available context
    retrieved from a PDF file along with page_contents and page number.

    You should only answer the user based on the following context and navigate the user
    to open the right page number to know more.

    Context:
    {context}
"""

# Chat with bot
def chat_with_bot(system_prompt, query):
    try:
        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            system_instruction=system_prompt
        )
        
        chat = model.start_chat(history=[])
        response = chat.send_message(query)
        return response.text
    except Exception as e:
        st.error(f"Error generating response: {e}")
        return "Sorry, I encountered an error while processing your query."

# Main Streamlit App
def main():
    st.title("📘 PDF Chat Assistant")
    st.markdown("Upload a PDF and chat with it using AI!")
    
    # Check for environment variables
    if not all([os.getenv("GOOGLE_API_KEY"), os.getenv("QDRANT_URL"), os.getenv("QDRANT_API_KEY")]):
        st.error("⚠️ Please set up your environment variables (GOOGLE_API_KEY, QDRANT_URL, QDRANT_API_KEY)")
        st.stop()
    
    # Initialize services
    if not configure_google_ai():
        st.stop()
    
    qdrant_client = setup_qdrant()
    if not qdrant_client:
        st.stop()
    
    # Sidebar for PDF upload and settings
    with st.sidebar:
        st.header("📁 PDF Upload")
        
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type="pdf",
            help="Upload a PDF file to start chatting"
        )
        
        if uploaded_file is not None:
            st.success(f"📄 {uploaded_file.name} uploaded successfully!")
            
            # Processing settings
            st.header("⚙️ Processing Settings")
            chunk_size = st.slider("Chunk Size", 500, 2000, 1000, 100)
            chunk_overlap = st.slider("Chunk Overlap", 100, 800, 400, 50)
            
            # Add batch size setting
            st.subheader("🔧 Advanced Settings")
            batch_size = st.selectbox("Batch Size", [25, 50, 100], index=1, help="Smaller batches help avoid timeouts")
            
            if st.button("🔄 Process PDF", type="primary"):
                with st.spinner("Processing PDF... This may take a few minutes."):
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        temp_path = tmp_file.name
                    
                    try:
                        # Load and process PDF
                        docs = load_pdf_as_documents(temp_path)
                        if docs:
                            st.info(f"📄 Loaded {len(docs)} pages from PDF")
                            chunks = split_documents(docs, chunk_size, chunk_overlap)
                            st.info(f"✂️ Split into {len(chunks)} chunks")
                            
                            # Index with retry mechanism
                            vector_store = index_documents_with_retry(
                                chunks, 
                                st.session_state.collection_name, 
                                qdrant_client
                            )
                            
                            if vector_store:
                                st.session_state.pdf_processed = True
                                st.session_state.vector_store = vector_store
                                st.balloons()
                            else:
                                st.error("❌ Failed to index documents")
                    finally:
                        # Clean up temporary file
                        os.unlink(temp_path)
        
        # Display processing status
        if st.session_state.pdf_processed:
            st.success("✅ PDF Ready for Chat!")
        else:
            st.info("📋 Upload and process a PDF to start chatting")
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
            
        # Reset PDF processing button
        if st.session_state.pdf_processed:
            if st.button("🔄 Reset PDF Processing"):
                st.session_state.pdf_processed = False
                st.session_state.vector_store = None
                st.session_state.messages = []
                st.rerun()
    
    # Main chat interface
    if st.session_state.pdf_processed:
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask me anything about the PDF..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate assistant response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    # Search for relevant context
                    results = search_query(prompt, st.session_state.collection_name, qdrant_client)
                    
                    if results:
                        # Build context and get response
                        system_prompt = build_prompt_context(results)
                        response = chat_with_bot(system_prompt, prompt)
                        
                        st.markdown(response)
                        
                        # Add assistant response to chat history
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        
                        # Show source information in expander
                        with st.expander("📚 Source Information"):
                            for i, result in enumerate(results[:3]):  # Show top 3 results
                                st.write(f"**Source {i+1}:**")
                                st.write(f"- Page: {result.payload['metadata']['page_label']}")
                                # st.write(f"- Relevance Score: {result.score:.3f}")
                                st.write(f"- Preview: {result.payload['page_content'][:200]}...")
                                st.divider()
                    else:
                        error_msg = "Sorry, I couldn't find relevant information in the PDF for your query."
                        st.markdown(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
    
    else:
        # Welcome message when no PDF is processed
        st.info("👈 Please upload and process a PDF file from the sidebar to start chatting!")
        
        # Sample queries
        st.markdown("### 💡 Example Questions You Can Ask:")
        st.markdown("""
        - "What is the main topic of this document?"
        - "Summarize the key points from page 5"
        - "Find information about [specific topic]"
        - "What are the conclusions mentioned in the document?"
        """)

if __name__ == "__main__":
    main()