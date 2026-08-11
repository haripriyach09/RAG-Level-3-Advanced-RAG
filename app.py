import os
import pandas as pd
import plotly.express as px
import streamlit as st

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.documents import Document

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from config import GROQ_API_KEY


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="RAG Level 3",
    page_icon="🧠",
    layout="wide"
)


# ============================================================
# CONFIGURATION
# ============================================================

DB_PATH = "chroma_db"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

LLM_MODEL = "llama-3.1-8b-instant"

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

INITIAL_TOP_K = 10

FINAL_TOP_K = 3

KEYWORD_TOP_K = 10


# ============================================================
# CONVERSATIONAL MEMORY
# ============================================================

if "chat_history" not in st.session_state:

    st.session_state.chat_history = []


# ============================================================
# LOAD EMBEDDINGS
# ============================================================

@st.cache_resource
def load_embeddings():

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )


# ============================================================
# LOAD VECTOR DATABASE
# ============================================================

@st.cache_resource
def load_vector_db():

    embeddings = load_embeddings()

    db = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings
    )

    return db


# ============================================================
# LOAD LLM
# ============================================================

@st.cache_resource
def load_llm():

    return ChatGroq(
        model=LLM_MODEL,
        api_key=GROQ_API_KEY,
        temperature=0
    )


# ============================================================
# LOAD RERANKER
# ============================================================

@st.cache_resource
def load_reranker():

    return CrossEncoder(
        RERANKER_MODEL
    )


# ============================================================
# LOAD ALL DOCUMENTS
# ============================================================

@st.cache_resource
def load_all_documents():

    db = load_vector_db()

    data = db.get()

    documents = data.get(
        "documents",
        []
    )

    metadatas = data.get(
        "metadatas",
        []
    )

    return documents, metadatas


# ============================================================
# CREATE BM25
# ============================================================

@st.cache_resource
def create_bm25():

    documents, metadatas = load_all_documents()

    tokenized_documents = [
        document.lower().split()
        for document in documents
    ]

    bm25 = BM25Okapi(
        tokenized_documents
    )

    return (
        bm25,
        documents,
        metadatas
    )


# ============================================================
# CONVERSATION HISTORY
# ============================================================

def get_chat_history():

    if not st.session_state.chat_history:

        return "No previous conversation."

    history = []

    for message in st.session_state.chat_history:

        if message["role"] == "human":

            history.append(
                f"Human: {message['content']}"
            )

        else:

            history.append(
                f"Assistant: {message['content']}"
            )

    return "\n".join(history)


# ============================================================
# HYBRID RETRIEVAL
# ============================================================

def hybrid_retrieval(question):

    db = load_vector_db()

    bm25, documents, metadatas = create_bm25()

    # --------------------------------------------------------
    # VECTOR SEARCH - TOP 10
    # --------------------------------------------------------

    vector_docs = db.similarity_search(
        question,
        k=INITIAL_TOP_K
    )

    # --------------------------------------------------------
    # BM25 KEYWORD SEARCH - TOP 10
    # --------------------------------------------------------

    tokenized_query = question.lower().split()

    bm25_results = bm25.get_top_n(
        tokenized_query,
        documents,
        n=KEYWORD_TOP_K
    )

    # --------------------------------------------------------
    # COMBINE RESULTS
    # --------------------------------------------------------

    combined = []

    seen = set()

    # Vector results
    for doc in vector_docs:

        text = doc.page_content

        if text not in seen:

            combined.append(doc)

            seen.add(text)

    # BM25 results
    for text in bm25_results:

        if text not in seen:

            index = documents.index(text)

            doc = Document(
                page_content=text,
                metadata=(
                    metadatas[index]
                    if metadatas[index]
                    else {}
                )
            )

            combined.append(doc)

            seen.add(text)

    return combined


# ============================================================
# RERANK DOCUMENTS
# ============================================================

def rerank_documents(
    question,
    documents
):

    reranker = load_reranker()

    if not documents:

        return []

    pairs = [
        [
            question,
            document.page_content
        ]
        for document in documents
    ]

    scores = reranker.predict(
        pairs
    )

    ranked_documents = sorted(
        zip(documents, scores),
        key=lambda x: x[1],
        reverse=True
    )

    final_documents = []

    for document, score in ranked_documents[
        :FINAL_TOP_K
    ]:

        document.metadata[
            "rerank_score"
        ] = float(score)

        final_documents.append(
            document
        )

    return final_documents


# ============================================================
# ASK QUESTION
# ============================================================

def ask_question(question):

    llm = load_llm()

    # --------------------------------------------------------
    # STEP 1: HYBRID RETRIEVAL
    # --------------------------------------------------------

    retrieved_documents = hybrid_retrieval(
        question
    )

    # --------------------------------------------------------
    # STEP 2: RERANK
    # --------------------------------------------------------

    reranked_documents = rerank_documents(
        question,
        retrieved_documents
    )

    # --------------------------------------------------------
    # CONTEXT
    # --------------------------------------------------------

    context = "\n\n".join(
        document.page_content
        for document in reranked_documents
    )

    # --------------------------------------------------------
    # MEMORY
    # --------------------------------------------------------

    conversation = get_chat_history()

    # --------------------------------------------------------
    # PROMPT
    # --------------------------------------------------------

    prompt = f"""
You are a helpful PDF question-answering assistant.

Use ONLY the retrieved context to answer the question.

Previous Conversation:
{conversation}

Retrieved Context:
{context}

Current Question:
{question}

Instructions:

1. Use the retrieved context whenever possible.
2. Use the previous conversation for follow-up questions.
3. Do not invent information.
4. If the answer is not available, say so.
5. Give a concise and accurate answer.
"""

    response = llm.invoke(
        prompt
    )

    return (
        response.content,
        reranked_documents,
        len(reranked_documents)
    )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "🧠 RAG Level 3"
)

st.sidebar.success(
    "✅ Task 1 - Conversational Memory"
)

st.sidebar.success(
    "✅ Task 2 - Hybrid Retrieval"
)

st.sidebar.success(
    "✅ Task 3 - Reranking"
)

st.sidebar.markdown("---")

st.sidebar.markdown(
    f"""
### Configuration

**Embedding**

`{EMBEDDING_MODEL}`

**LLM**

`{LLM_MODEL}`

**Initial Retrieval**

Top `{INITIAL_TOP_K}`

**Reranker**

`{RERANKER_MODEL}`

**Final Results**

Top `{FINAL_TOP_K}`
"""
)

if st.sidebar.button(
    "🗑️ Clear Conversation"
):

    st.session_state.chat_history = []

    st.rerun()


# ============================================================
# HEADER
# ============================================================

st.title(
    "🧠 RAG Level 3 - Advanced RAG"
)

st.caption(
    "Conversational Memory | "
    "Hybrid Retrieval | "
    "Reranking"
)

st.info(
    """
The system retrieves documents using Vector Search
and BM25 Keyword Search, then reranks the retrieved
documents before sending the best results to the LLM.
"""
)


# ============================================================
# RETRIEVAL PIPELINE
# ============================================================

st.subheader(
    "🔎 Retrieval Pipeline"
)

st.code(
    """
Question
   ↓
Hybrid Retrieval
   ↓
Top 10
   ↓
Reranker
   ↓
Top 3
   ↓
LLM
""",
    language="text"
)


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

if st.session_state.chat_history:

    st.subheader(
        "💬 Conversation"
    )

    for message in st.session_state.chat_history:

        if message["role"] == "human":

            with st.chat_message("user"):

                st.write(
                    message["content"]
                )

        else:

            with st.chat_message("assistant"):

                st.write(
                    message["content"]
                )


# ============================================================
# CHAT INPUT
# ============================================================

question = st.chat_input(
    "Ask a question about the PDFs..."
)


# ============================================================
# PROCESS QUESTION
# ============================================================

if question:

    with st.chat_message("user"):

        st.write(question)

    with st.chat_message("assistant"):

        with st.spinner(
            "Retrieving and reranking..."
        ):

            try:

                (
                    answer,
                    retrieved_docs,
                    final_count
                ) = ask_question(
                    question
                )

                st.write(
                    answer
                )

            except Exception as e:

                answer = "An error occurred."

                retrieved_docs = []

                final_count = 0

                st.error(str(e))

    # --------------------------------------------------------
    # SAVE MEMORY
    # --------------------------------------------------------

    st.session_state.chat_history.append(
        {
            "role": "human",
            "content": question
        }
    )

    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

   # --------------------------------------------------------
    # RERANKED SOURCES
    # --------------------------------------------------------

    if retrieved_docs:

        with st.expander(
            "🏆 Top 3 Reranked Results",
            expanded=True
        ):

            st.write(
                f"Final documents sent to LLM: "
                f"**{final_count}**"
            )

            for i, doc in enumerate(
                retrieved_docs[:FINAL_TOP_K],
                start=1
            ):

                source = doc.metadata.get(
                    "source",
                    "Unknown source"
                )

                page = doc.metadata.get(
                    "page",
                    "Unknown page"
                )

                score = doc.metadata.get(
                    "rerank_score",
                    0
                )

                st.markdown(
                    f"""
    ### Result {i}

    📄 **Source:** `{os.path.basename(source)}`

    📑 **Page:** `{page}`

    🎯 **Reranker Score:** `{score:.4f}`
    """
                )
# ============================================================
# TASK 4 - RETRIEVAL EVALUATION
# ============================================================

st.sidebar.success("🧪 Task 4 - Retrieval Evaluation")

# ============================================================
# EVALUATION DATASET
# ============================================================

evaluation_data = [
    {
        "Question": "What is the name of the project?",
        "Expected Answer": "Kahaani"
    },
    {
        "Question": "What database is mentioned in the project?",
        "Expected Answer": "SQLite database"
    },
    {
        "Question": "What algorithm handles user feedback?",
        "Expected Answer": "Feedback Handling Algorithm"
    },
    {
        "Question": "What does the feedback algorithm improve?",
        "Expected Answer": "personalization and recommendation patterns"
    },
    {
        "Question": "How is user information protected?",
        "Expected Answer": "anonymization and privacy protection"
    },
    {
        "Question": "What type of user input does the system collect?",
        "Expected Answer": "text prompts symptom descriptions emotional reflections preferred story genres"
    },
    {
        "Question": "Where is structured user data stored?",
        "Expected Answer": "SQLite database"
    },
    {
        "Question": "What happens to user feedback?",
        "Expected Answer": "ratings comments and reactions are recorded"
    },
    {
        "Question": "What is the purpose of data organization and maintenance?",
        "Expected Answer": "maintain data integrity"
    },
    {
        "Question": "What does the system use feedback for?",
        "Expected Answer": "refining templates improving personalization future recommendations"
    }
]


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text):

    return set(
        text.lower()
        .replace(",", "")
        .replace(".", "")
        .replace(":", "")
        .replace(";", "")
        .replace("(", "")
        .replace(")", "")
        .split()
    )


# ============================================================
# PRECISION
# ============================================================

def calculate_precision(expected, retrieved):

    expected_words = normalize_text(expected)

    retrieved_words = normalize_text(retrieved)

    if not retrieved_words:
        return 0.0

    matched = expected_words.intersection(
        retrieved_words
    )

    return len(matched) / len(retrieved_words)


# ============================================================
# RECALL
# ============================================================

def calculate_recall(expected, retrieved):

    expected_words = normalize_text(expected)

    retrieved_words = normalize_text(retrieved)

    if not expected_words:
        return 0.0

    matched = expected_words.intersection(
        retrieved_words
    )

    return len(matched) / len(expected_words)


# ============================================================
# RETRIEVAL ACCURACY
# ============================================================

def calculate_retrieval_accuracy(
    expected,
    retrieved
):

    expected_words = normalize_text(expected)

    retrieved_words = normalize_text(retrieved)

    if not expected_words:
        return 0.0

    matched = expected_words.intersection(
        retrieved_words
    )

    score = len(matched) / len(expected_words)

    return 1.0 if score >= 0.5 else 0.0


# ============================================================
# ANSWER CORRECTNESS
# ============================================================

def calculate_answer_correctness(
    expected,
    answer
):

    expected_words = normalize_text(expected)

    answer_words = normalize_text(answer)

    if not expected_words:
        return 0.0

    matched = expected_words.intersection(
        answer_words
    )

    return len(matched) / len(expected_words)


# ============================================================
# TASK 4 DASHBOARD
# ============================================================

st.divider()

st.header(
    "🧪 Task 4 - Retrieval Evaluation"
)

st.write(
    "Evaluate the RAG system using 10 predefined "
    "questions and expected answers."
)

st.info(
    """
The evaluation measures whether the expected information
is retrieved and whether the generated answer contains
the expected information.
"""
)


# ============================================================
# EVALUATION QUESTIONS
# ============================================================

st.subheader(
    "📋 Evaluation Questions"
)

evaluation_df = pd.DataFrame(
    evaluation_data
)

st.dataframe(
    evaluation_df,
    width="stretch",
    hide_index=True
)


# ============================================================
# RUN EVALUATION
# ============================================================

if st.button(
    "🚀 Run Retrieval Evaluation",
    type="primary"
):

    evaluation_results = []

    progress = st.progress(0)

    total_questions = len(
        evaluation_data
    )

    for index, item in enumerate(
        evaluation_data
    ):

        question = item["Question"]

        expected = item["Expected Answer"]

        try:

            # ------------------------------------------------
            # RUN EXISTING RAG PIPELINE
            # ------------------------------------------------

            (
                answer,
                retrieved_docs,
                final_count
            ) = ask_question(question)


            # ------------------------------------------------
            # BUILD RETRIEVED CONTEXT
            # ------------------------------------------------

            retrieved_context = "\n".join(
                document.page_content
                for document in retrieved_docs
            )


            # ------------------------------------------------
            # CALCULATE METRICS
            # ------------------------------------------------

            retrieval_accuracy = (
                calculate_retrieval_accuracy(
                    expected,
                    retrieved_context
                )
            )

            precision = calculate_precision(
                expected,
                retrieved_context
            )

            recall = calculate_recall(
                expected,
                retrieved_context
            )

            answer_correctness = (
                calculate_answer_correctness(
                    expected,
                    answer
                )
            )


            # ------------------------------------------------
            # SAVE RESULT
            # ------------------------------------------------

            evaluation_results.append(
                {
                    "Question": question,
                    "Expected Answer": expected,
                    "Generated Answer": answer,
                    "Retrieval Accuracy": retrieval_accuracy,
                    "Precision": precision,
                    "Recall": recall,
                    "Answer Correctness": answer_correctness
                }
            )


        except Exception as e:

            evaluation_results.append(
                {
                    "Question": question,
                    "Expected Answer": expected,
                    "Generated Answer": "Error",
                    "Retrieval Accuracy": 0.0,
                    "Precision": 0.0,
                    "Recall": 0.0,
                    "Answer Correctness": 0.0
                }
            )


        progress.progress(
            (index + 1) / total_questions
        )


    # --------------------------------------------------------
    # SAVE RESULTS IN SESSION
    # --------------------------------------------------------

    st.session_state[
        "evaluation_results"
    ] = evaluation_results


# ============================================================
# DISPLAY EVALUATION RESULTS
# ============================================================

if "evaluation_results" in st.session_state:

    results = st.session_state[
        "evaluation_results"
    ]

    results_df = pd.DataFrame(
        results
    )


    # ========================================================
    # CONVERT TO PERCENTAGES
    # ========================================================

    metric_columns = [
        "Retrieval Accuracy",
        "Precision",
        "Recall",
        "Answer Correctness"
    ]

    for column in metric_columns:

        results_df[column] = (
            results_df[column] * 100
        ).round(2)


    # ========================================================
    # OVERALL METRICS
    # ========================================================

    st.subheader(
        "📊 Overall Evaluation Metrics"
    )

    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "Retrieval Accuracy",
            f"{results_df['Retrieval Accuracy'].mean():.2f}%"
        )


    with col2:

        st.metric(
            "Precision",
            f"{results_df['Precision'].mean():.2f}%"
        )


    with col3:

        st.metric(
            "Recall",
            f"{results_df['Recall'].mean():.2f}%"
        )


    with col4:

        st.metric(
            "Answer Correctness",
            f"{results_df['Answer Correctness'].mean():.2f}%"
        )


    # ========================================================
    # RESULTS TABLE
    # ========================================================

    st.subheader(
        "📈 Detailed Evaluation Results"
    )

    display_results = results_df[
        [
            "Question",
            "Retrieval Accuracy",
            "Precision",
            "Recall",
            "Answer Correctness"
        ]
    ]

    st.dataframe(
        display_results,
        width="stretch",
        hide_index=True
    )


    # ========================================================
    # METRIC CHART
    # ========================================================

    st.subheader(
        "📊 Evaluation Metrics Comparison"
    )

    average_metrics = {
        "Metric": [
            "Retrieval Accuracy",
            "Precision",
            "Recall",
            "Answer Correctness"
        ],
        "Score": [
            results_df[
                "Retrieval Accuracy"
            ].mean(),

            results_df[
                "Precision"
            ].mean(),

            results_df[
                "Recall"
            ].mean(),

            results_df[
                "Answer Correctness"
            ].mean()
        ]
    }

    metric_df = pd.DataFrame(
        average_metrics
    )

    fig = px.bar(
        metric_df,
        x="Metric",
        y="Score",
        text="Score",
        title="Overall RAG Evaluation"
    )

    fig.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside"
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )


    # ========================================================
    # GENERATED ANSWERS
    # ========================================================

    st.subheader(
        "🤖 Generated Answers"
    )

    for index, row in results_df.iterrows():

        with st.expander(
            f"Question {index + 1}: "
            f"{row['Question']}"
        ):

            st.markdown(
                "### Expected Answer"
            )

            st.write(
                row["Expected Answer"]
            )


            st.markdown(
                "### Generated Answer"
            )

            st.write(
                row["Generated Answer"]
            )


            col1, col2, col3, col4 = st.columns(4)


            with col1:

                st.metric(
                    "Retrieval Accuracy",
                    f"{row['Retrieval Accuracy']:.2f}%"
                )


            with col2:

                st.metric(
                    "Precision",
                    f"{row['Precision']:.2f}%"
                )


            with col3:

                st.metric(
                    "Recall",
                    f"{row['Recall']:.2f}%"
                )


            with col4:

                st.metric(
                    "Answer Correctness",
                    f"{row['Answer Correctness']:.2f}%"
                )