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


git clone https://github.com/your-username/loan-rag-chatbot.git
cd loan-rag-chatbot

✅ Install Dependencies
pip install -r requirements.txt

✅ Add Dataset
loan_rag_chatbot/
├── loan_data/
│   ├── Training Dataset.csv
│   ├── Test Dataset.csv
│   └── Sample_Submission.csv

💡 Usage
Run the chatbot:
python app.py
🤖 Loan Chatbot is ready! Ask any question about the loan dataset.
You: How many dependents does LP001003 have?
Bot: 3+

You: What is the income of the applicant with Loan ID LP001005?
Bot: 6000

🧠 Tech Stack
Component	Tech Used
Embedding Model	sentence-transformers/all-MiniLM-L6-v2
Vector Search Engine	FAISS
Generative LLM	google/flan-t5-base (Hugging Face)
Programming Language	Python

🔍 Example Questions to Try
What is the loan amount of LP001178?

How many dependents does LP002837 have?

Is LP001003 a self-employed applicant?

What is the loan status of LP001100?

🧑‍💻 Author
Built with ❤️ by Ronak.
Feel free to fork, modify, or use this for learning and projects!

