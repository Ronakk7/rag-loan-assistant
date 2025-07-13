# 💬 Loan RAG Chatbot

A lightweight **Retrieval-Augmented Generation (RAG)** based Q&A chatbot that answers natural language questions from a structured **loan approval dataset** using document retrieval and a generative model. Built using **FAISS**, **SentenceTransformers**, and Hugging Face's **Flan-T5**, this project demonstrates how traditional tabular data can be used with modern generative AI for intelligent query answering.

---

## 📊 Dataset

This chatbot uses the publicly available **Loan Approval Prediction Dataset** from Kaggle:

🔗 [Loan Approval Prediction Dataset on Kaggle](https://www.kaggle.com/datasets/sonalisingh1411/loan-approval-prediction)

It includes structured fields like:
- Loan ID, Gender, Marital Status, Education
- Income, Loan Amount, Credit History
- Loan Approval Status

---

## ⚙️ How It Works

1. **Tabular Data → Text Rows:** Each row from the CSV is converted to a readable text format (e.g. `Loan_ID: LP001003 | Gender: Male | ...`).
2. **Embedding & Indexing:** All rows are embedded using SentenceTransformers and indexed using FAISS for fast similarity search.
3. **Context Retrieval:** For any natural language question, the top `k` most relevant rows are retrieved.
4. **Answer Generation:** A generative model (`flan-t5-base`) is prompted with the retrieved rows to generate a precise answer.

---

## 🚀 Quickstart

### ✅ Clone the Repository
```bash
git clone https://github.com/your-username/loan-rag-chatbot.git
cd loan-rag-chatbot
