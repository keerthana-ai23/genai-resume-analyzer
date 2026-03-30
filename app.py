import streamlit as st
import os
import tempfile
from dotenv import load_dotenv

# Load API key
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

st.title("🤖 AI Resume Analyzer")

# Upload resume
uploaded_file = st.file_uploader("Upload your resume (PDF)", type="pdf")

# Job description input
job_description = st.text_area("📌 Paste Job Description")

docs = []

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_path = tmp_file.name

    loader = PyPDFLoader(temp_path)
    documents = loader.load()

    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(documents)

    st.subheader("📄 Resume Content:")
    for doc in docs:
        st.write(doc.page_content)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectors = embeddings.embed_documents([doc.page_content for doc in docs])

    st.subheader("🧠 Embeddings Created:")
    st.write(f"Chunks: {len(vectors)}")

# Query input
st.subheader("💬 Ask Questions / Analyze Resume")
query = st.text_input("Ask something:")

if query and docs and job_description:
    llm = ChatOpenAI(model="gpt-4o-mini")

    context = "\n".join([doc.page_content for doc in docs])

  prompt = f"""
You are an AI resume evaluator.

Compare the resume with the job description.

Resume:
{context}

Job Description:
{job_description}

Return output in this format:

Match Score: XX%

Missing Skills:
- skill1
- skill2

Suggestions:
- suggestion1
- suggestion2
"""

    response = llm.invoke(prompt)

    st.subheader("📊 Resume Analysis:")
    st.write(response.content)
