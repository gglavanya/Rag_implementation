

# 📄 PDF Intelligence Scout: Local RAG with Semantic Caching

A privacy-first, high-performance RAG (Retrieval-Augmented Generation) application designed to turn complex documents into interactive intelligence. This system runs entirely on your local machine, ensuring data privacy while maintaining lightning-fast response times through a custom semantic caching layer.

## 🚀 Key Features

* **100% Local Inference:** Utilizes **Ollama (Llama 3.2)** and **HuggingFace Embeddings** to process data without cloud dependencies.
* **⚡ Semantic Caching:** Implements a vector-based cache using **ChromaDB** to detect semantically similar questions, reducing latency for repeat queries from seconds to milliseconds.
* **🔄 Dynamic Data Ingestion:** Features a real-time "hot-reload" pipeline. When a new PDF is uploaded via the **Streamlit** interface, the system automatically handles re-chunking and vector database updates.
* **🛡️ Hallucination Guardrails:** Optimized prompt templates and retrieval parameters (Top-K) to ensure the AI remains strictly grounded in the source document.
* **📊 Automated Project Mapping:** Specifically tuned to correlate technical project details with professional role responsibilities.

---

## 🛠️ Technical Stack

* **Orchestration:** [LangChain](https://www.langchain.com/)
* **LLM:** [Ollama](https://ollama.com/) (Llama 3.2:1b)
* **Vector Database:** [ChromaDB](https://www.trychroma.com/)
* **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`
* **UI Framework:** [Streamlit](https://streamlit.io/)
* **Language:** Python 3.10+

---

## 📂 Project Structure

```text
├── app.py              # Main Streamlit application & UI logic
├── utils.py            # Backend logic: Semantic cache, Chunking, & Ingestion
├── vector_db/          # Persistent storage for document embeddings
├── my_semantic_cache/  # Persistent storage for Q&A pairs
└── Rag_docs.pdf        # Source document (Resume/Portfolio)

```

---

## ⚙️ Setup & Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/Rag_implementation.git
cd Rag_implementation

```


2. **Install dependencies:**
```bash
pip install streamlit langchain langchain-community langchain-ollama langchain-chroma langchain-huggingface pypdf

```


3. **Install & Run Ollama:**
* Download Ollama from [ollama.com](https://ollama.com/).
* Pull the model: `ollama pull llama3.2:1b`


4. **Launch the App:**
```bash
streamlit run app.py

```



---

## 🧠 System Logic: How it Works

1. **Input:** The user asks a question through the Streamlit chat interface.
2. **Cache Check:** The system checks `my_semantic_cache` for a similar previous question (Threshold: 0.90 similarity).
3. **Retrieval:** If a cache miss occurs, the system retrieves the top 5 relevant text chunks from the PDF using ChromaDB.
4. **Generation:** The context and question are sent to the local Llama 3.2 model with strict grounding instructions.
5. **Update:** The new answer is saved to the Semantic Cache for future instant retrieval.

---

## 📈 Future Enhancements

* [ ] Implementation of **Multi-Query Retrieval** to handle complex questions.
* [ ] Integration of a **Reranker** (Cross-Encoder) for even higher accuracy.
* [ ] Support for multi-document comparison.



