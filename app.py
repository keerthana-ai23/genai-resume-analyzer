import streamlit as st
import os
import tempfile
import re
from dotenv import load_dotenv

# Load API key
load_dotenv()

# Imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS

# UI Title
st.title("🤖 AI Resume Analyzer (RAG + Scoring)")

# Upload PDF
uploaded_file = st.file_uploader("Upload your resume (PDF)", type="pdf")

if uploaded_file is not None:

    # Save file temporarily
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_path = tmp_file.name

    # Load PDF
    loader = PyPDFLoader(temp_path)
    documents = loader.load()

    # Split text
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(documents)

    st.subheader("📄 Resume Preview")
    st.write(docs[0].page_content[:500] + "...")

    # -------------------------------
    # 🔥 RAG Setup
    # -------------------------------
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(docs, embeddings)
    retriever = db.as_retriever()

    st.success("✅ RAG Ready (Embeddings + Retrieval)")

    # -------------------------------
    # 💬 Ask Questions
    # -------------------------------
    st.subheader("💬 Ask Questions About Your Resume")

    query = st.text_input("Ask something:")

    if query:
        llm = ChatOpenAI(model="gpt-3.5-turbo")

        # Retrieve relevant chunks
        relevant_docs = retriever.invoke(query)
        context = "\n".join([doc.page_content for doc in relevant_docs])

        prompt = f"""
You are an AI resume assistant.

Answer based only on the context below.

Context:
{context}

Question:
{query}
"""

        response = llm.invoke(prompt)

        st.subheader("🤖 Answer")
        st.write(response.content)

    # -------------------------------
    # 📄 Job Description Matching
    # -------------------------------
    st.subheader("📄 Paste Job Description")

    jd = st.text_area("Paste job description here:")

    if jd:
        llm = ChatOpenAI(model="gpt-3.5-turbo")

        # Retrieve relevant chunks for JD
        relevant_docs = retriever.invoke(jd)
        context = "\n".join([doc.page_content for doc in relevant_docs])

        prompt = f"""
You are an AI resume evaluator.

Compare the resume with the job description.

Resume Context:
{context}

Job Description:
{jd}

Return output EXACTLY in this format:

Match Score: XX%

Missing Skills:
- skill1
- skill2

Suggestions:
- suggestion1
- suggestion2
"""

        response = llm.invoke(prompt)

        result = response.content

        st.subheader("📊 JD Match Analysis")
        st.write(result)

        # -------------------------------
        # 🔥 Extract REAL Score
        # -------------------------------
        match = re.search(r"(\d+)%", result)

        if match:
            score = int(match.group(1))
        else:
            score = 50  # fallback

        # -------------------------------
        # 📊 Display Score
        # -------------------------------
        st.subheader("📊 Match Score")

        st.progress(score / 100)
        st.metric("Match Score", f"{score}%")
