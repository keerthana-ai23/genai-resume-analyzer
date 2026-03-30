import streamlit as st
import os
import tempfile
from dotenv import load_dotenv

# Load API key from .env
load_dotenv()

# Imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# UI Title
st.title("🤖 AI Resume Analyzer")

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

    st.subheader("📄 Resume Content:")
    for doc in docs:
        st.write(doc.page_content)

    # Create embeddings
    embeddings = OpenAIEmbeddings()
    vectors = embeddings.embed_documents([doc.page_content for doc in docs])

    st.subheader("🧠 Embeddings Created:")
    st.write(f"Number of chunks: {len(vectors)}")
    st.write(f"Vector size: {len(vectors[0])}")

    # 💬 Question input (RAG)
    st.subheader("💬 Ask Questions About Your Resume")
    query = st.text_input("Ask something:")

    if query:
        llm = ChatOpenAI(model="gpt-3.5-turbo")

        # Combine all resume text
        context = "\n".join([doc.page_content for doc in docs])

        prompt = f"""
        You are an AI resume assistant.

        Answer the question based only on the resume below.

        Resume:
        {context}

        Question:
        {query}
        """

        response = llm.invoke(prompt)

        st.subheader("🤖 Answer:")
        st.write(response.content)
