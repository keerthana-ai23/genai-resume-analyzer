# 🚀 Gen AI Resume Analyzer (RAG-based)

This project is a Generative AI-powered Resume Analyzer that evaluates resumes against job descriptions using Retrieval-Augmented Generation (RAG).

## 🔍 Features
- Upload resume (PDF)
- Compare with job description
- Extract skills and match score
- Identify missing skills
- Generate AI-powered feedback

## ⚙️ Tech Stack
- Python
- LangChain
- OpenAI API
- FAISS (Vector Database)
- Streamlit

## 🧠 How it Works
1. Resume is parsed and split into chunks  
2. Embeddings are created using OpenAI  
3. Stored in FAISS vector database  
4. Retrieved context is passed to LLM  
5. AI generates insights and feedback  

## ▶️ Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
