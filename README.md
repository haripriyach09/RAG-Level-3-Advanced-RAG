# 🧠 RAG Level 3 - Advanced RAG

An advanced Retrieval-Augmented Generation (RAG) system built with Python, LangChain, Streamlit, ChromaDB, BM25, Sentence Transformers, and Groq LLM.

This project extends the previous RAG levels by introducing **conversational memory, hybrid retrieval, document reranking, and retrieval evaluation**.

---

## 🚀 Features

### 🧠 Task 1 - Conversational Memory

The chatbot maintains conversation history using Streamlit session state.

This allows the system to understand follow-up questions based on previous interactions.

### 🔎 Task 2 - Hybrid Retrieval

Instead of relying only on vector similarity search, the system combines:

* Vector Search
* BM25 Keyword Search

The results from both retrieval methods are combined before reranking.

### 🎯 Task 3 - Reranking

The retrieved documents are reranked using a Cross-Encoder.

The pipeline is:

```text
Question
   ↓
Hybrid Retrieval
   ↓
Initial Retrieval
   ↓
Reranker
   ↓
Top 3
   ↓
LLM
   ↓
Answer
```

### 🧪 Task 4 - Retrieval Evaluation

The system evaluates the RAG pipeline using 10 predefined questions and expected answers.

The following metrics are calculated:

* Retrieval Accuracy
* Precision
* Recall
* Answer Correctness

The dashboard displays both individual question results and overall evaluation scores.

---

## ⚙️ Technology Stack

| Technology            | Purpose                    |
| --------------------- | -------------------------- |
| Python                | Core programming           |
| Streamlit             | Web dashboard              |
| LangChain             | RAG framework              |
| ChromaDB              | Vector database            |
| Sentence Transformers | Embeddings                 |
| BM25                  | Keyword retrieval          |
| Cross-Encoder         | Document reranking         |
| Groq                  | LLM inference              |
| Pandas                | Evaluation data processing |
| Plotly                | Result visualization       |

---

## 🤖 Models Used

### Embedding Model

```text
sentence-transformers/all-MiniLM-L6-v2
```

### LLM

```text
llama-3.1-8b-instant
```

### Reranker

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

---

## 📂 Project Structure

```text
RAG_LEVEL3/
│
├── app.py
├── ingest.py
├── config.py
├── requirements.txt
├── .gitignore
│
├── loaders/
│   └── ...
│
├── services/
│   └── ...
│
├── pdfs/
│   ├── FINAL_SYNOPSIS.pdf
│   └── kahaanifinalreport.pdf
│
├── chroma_db/
│
└── myenv/
```

> `myenv/`, `.env`, and generated vector database files should not be committed to GitHub.

---

## 📚 Document Ingestion

The project processes multiple PDF documents and creates vector embeddings.

Example:

```text
FINAL_SYNOPSIS.pdf
kahaanifinalreport.pdf
```

The documents are split into chunks and stored in ChromaDB for retrieval.

---

## 🔄 RAG Pipeline

The complete pipeline is:

```text
                User Question
                      ↓
              Conversation Memory
                      ↓
               Hybrid Retrieval
                 ↙          ↘
          Vector Search     BM25
                 ↘          ↙
                  Combined Results
                        ↓
                    Reranker
                        ↓
                      Top 3
                        ↓
                       LLM
                        ↓
                     Answer
                        ↓
              Sources + Evaluation
```

---

## 📊 Evaluation

The evaluation dataset contains 10 questions with predefined expected answers.

The system calculates:

```text
Retrieval Accuracy
Precision
Recall
Answer Correctness
```

The Streamlit dashboard displays:

* Overall metric scores
* Detailed question-level results
* Generated answers
* Expected answers
* Evaluation charts

---

## 🖥️ Running the Project

### 1. Clone the repository

```bash
git clone https://github.com/haripriyach09/RAG-Level-3-Advanced-RAG.git
```

### 2. Open the project

```bash
cd RAG-Level-3-Advanced-RAG
```

### 3. Create a virtual environment

```bash
python -m venv myenv
```

### 4. Activate the environment

Windows:

```powershell
myenv\Scripts\activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Configure the API key

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key
```

Never upload the `.env` file to GitHub.

### 7. Create the vector database

```bash
python ingest.py
```

### 8. Run Streamlit

```bash
streamlit run app.py
```

The application will open at:

```text
http://localhost:8501
```

---

## 🎯 Project Objectives

* Implement conversational memory in RAG.
* Improve retrieval using hybrid search.
* Improve document relevance using reranking.
* Evaluate retrieval and answer quality quantitatively.
* Provide an interactive Streamlit dashboard.
* Build a complete advanced RAG pipeline.

---

## 👩‍💻 Project

**RAG Level 3 - Advanced RAG**

Built as part of a progressive RAG implementation covering:

```text
Level 1 → Model & Chunking Experiments
Level 2 → Multi-PDF RAG
Level 3 → Advanced RAG
```

---

