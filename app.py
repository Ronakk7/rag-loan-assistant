# app.py

import os
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# === Step 1: Load CSV ===
print("📄 Loading dataset...")
data_path = os.path.join("loan_data", "Training Dataset.csv")
df = pd.read_csv(data_path)

# === Step 2: Convert rows to text "documents" ===
print("🧠 Converting rows to text...")
docs = []
for _, row in df.iterrows():
    doc = " | ".join([f"{col}: {row[col]}" for col in df.columns])
    docs.append(doc)

# === Step 3: Generate embeddings ===
print("🔎 Generating embeddings...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
doc_embeddings = embedder.encode(docs)

# === Step 4: Build FAISS index ===
dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(doc_embeddings))

# === Step 5: Load Hugging Face model ===
print("🤖 Loading model (flan-t5-base)...")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
generator = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

# === Step 6: Helper functions ===
def retrieve_context(query, k=3):
    query_emb = embedder.encode([query])
    _, indices = index.search(np.array(query_emb), k)
    return [docs[i] for i in indices[0]]

def generate_response(query):
    context = retrieve_context(query, k=3)
    prompt = f"Answer the question using this data:\n{context}\n\nQuestion: {query}"
    output = generator(prompt, max_new_tokens=150)[0]['generated_text']
    return output.strip()

# === Step 7: Chat Loop ===
print("\n🤖 Loan Chatbot is ready! Ask any question about the loan dataset.")
print("Type 'exit' to quit.")

while True:
    user_input = input("\nYou: ")
    if user_input.lower() in ['exit', 'quit']:
        print("👋 Exiting chatbot. Goodbye!")
        break
    response = generate_response(user_input)
    print("Bot:", response)
