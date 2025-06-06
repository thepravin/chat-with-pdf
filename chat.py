from langchain_qdrant import QdrantVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()
genai.configure(api_key = os.getenv("GOOGLE_API_KEY"))

# -------------- Vector Embedding -------------

embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001"
)

#------------- Vector Db connection -----------
vector_db  = QdrantVectorStore.from_existing_collection(
     url = "http://localhost:6333",
    collection_name = "learning_vectors",
    embedding=embedding_model
)



# Take a user Query
query = input("👨 > ")


# Vector Similarity Search [query] in DB
search_results = vector_db.similarity_search(
    query=query
)

# print("\nSearch Result : \n",search_results)

context = "\n\n\n".join([f"Page Content: {result.page_content}\nPage Number: {result.metadata['page_label']}\nFile Location: {result.metadata['source']}" for result in search_results])

SYSTEM_PROMPT = f"""
     You are a helpfull AI Assistant who asnweres user query based on the available context
    retrieved from a PDF file along with page_contents and page number.

    You should only ans the user based on the following context and navigate the user
    to open the right page number to know more.

    Context:
    {context}

"""

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",  
    system_instruction=SYSTEM_PROMPT  
)

chat_history = []

while True:
    if query.lower() in ["exit", "quit","q"]:
        print("\n Buy ✌️  ✌️  ✌️ ...\n")
        break

    chat = model.start_chat(history=chat_history)
    response = chat.send_message(query)
    print(f"🤖 : {response.text}")
    query = input("👨 > ")