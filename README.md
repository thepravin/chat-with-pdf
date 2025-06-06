#  InsightPDF – Chat with Your PDFs using AI

InsightPDF is an intelligent, interactive PDF assistant powered by **Google Gemini** and **LangChain**. Upload any PDF document and ask natural language questions — from summaries to specific content — and receive accurate, context-aware responses with source references.

>  Powered by: `Streamlit`, `LangChain`, `Gemini API`, `Qdrant`, `GoogleGenerativeAIEmbeddings`

---

##  Features

* ✅ Upload any PDF file and extract content
* ✂️ Smart document chunking with adjustable chunk size and overlap
* 🧠 Index vector embeddings using Google Gemini Embeddings (`embedding-001`)
* 🗃️ Store and query vectors using Qdrant Vector Database
* 💬 Natural language Q\&A on your PDF using Gemini 1.5 Flash
* 🔍 Source tracing: Shows page numbers and content previews
* ♻️ Retry and batch-indexing logic for large PDFs
* 📤 Intuitive Streamlit UI with file upload, settings, and chat interface

---

## 🖇️ Live URL :
```

https://insight-pdf.streamlit.app/

```

---

## 📸 Demo




---

##  Tech Stack

| Component       | Technology                     |
| --------------- | ------------------------------ |
| Frontend        | Streamlit                      |
| LLM & Embedding | Google Gemini (GenerativeAI)   |
| Vector Store    | Qdrant                         |
| PDF Parsing     | LangChain + PyMuPDF            |
| Text Splitting  | LangChain's Recursive Splitter |
| Hosting         | Local / Streamlit Cloud        |

---

##  Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/InsightPDF.git
cd InsightPDF
```

### 2. Install Requirements

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create a `.env` file in the root directory with the following:

```env
GOOGLE_API_KEY=your_gemini_api_key
QDRANT_URL=https://your-qdrant-instance.com
QDRANT_API_KEY=your_qdrant_api_key
```

>  You can use [Qdrant Cloud](https://cloud.qdrant.io) for hosted vector DB.

---

### 4. Run the App

```bash
streamlit run main.py
```

Open your browser at `http://localhost:8501`.

---

##  How It Works

1. **Upload PDF** – Uses LangChain’s `PyPDFLoader` to extract page-wise content.
2. **Split Content** – Uses `RecursiveCharacterTextSplitter` to chunk large texts.
3. **Generate Embeddings** – Google Generative AI (`embedding-001`) converts chunks into vectors.
4. **Index in Qdrant** – Embeddings are stored in a Qdrant collection.
5. **Chat with AI** – User query is vectorized, matched with top results, and passed to Gemini Chat.
6. **Response** – The bot returns a natural answer, citing source pages and context.

---

##  Example Questions :

After uploading and processing a PDF, try asking:

* "What is the main topic of this document?"
* "Summarize page 10"
* "What are the key points related to deep learning?"
* "Does this PDF include a case study?"
* "List all references used"

---

##  Customization Options

You can tweak:

*  `chunk_size` and `chunk_overlap` via sidebar
*  Batch size for embedding indexing
*  LLM model (`gemini-1.5-flash`) or change system prompt logic
*  Embedding models (currently uses `embedding-001`)


---


## 📚 Dependencies

Install all via `pip install -r requirements.txt`

```text
streamlit
python-dotenv
google-generativeai
langchain
langchain-google-genai
langchain-community
qdrant-client
pymupdf
```


---

## 🙌 Contributing

Contributions are welcome! Please open an issue or submit a PR. Let’s build a better way to interact with documents!

---

## 📝 License

[MIT License](LICENSE)

---

## 👨‍💻 Author

**Pravin**
📧 [LinkedIn](https://www.linkedin.com/in/thepravin)
💼 Software Engineer | AI Builder | Full-Stack Developer


