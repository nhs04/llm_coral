import os
import re
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
from transformers import GPT2TokenizerFast as GPT2Tokenizer
import tensorflow as tf
from load_data import read_file

app = Flask(__name__)

# === Load GPT-2 TFLite Model ===
print("[INFO] Loading GPT-2 TFLite model...")
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
gpt2_interpreter = tf.lite.Interpreter(model_path="gpt2_model.tflite")
gpt2_interpreter.allocate_tensors()
input_details = gpt2_interpreter.get_input_details()
output_details = gpt2_interpreter.get_output_details()
print("[SUCCESS] GPT-2 TFLite model ready.")

# === Load Embedding Model ===
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# === Load Excel File using load_data.py ===
try:
    file_name = r"C:\Users\adam\Desktop\llm_coral\data\hospital_patients.xlsx"
    file_data = read_file(file_name)
    df = pd.concat([chunk for chunk in file_data], ignore_index=True) if not isinstance(file_data, pd.DataFrame) else file_data
    if df.empty:
        raise ValueError("Empty file!")
    df.columns = df.columns.astype(str)
    print("[INFO] File loaded with columns:", df.columns.tolist())
except Exception as e:
    print(f"[ERROR] Failed to load data: {e}")
    df = pd.DataFrame()

def diagnose_condition(user_input):
    if df.empty:
        return {"error": "Excel data not loaded."}
    if 'Diagnosis' not in df.columns:
        return {"error": "Diagnosis column not found in file."}

    diagnosis_pool = df['Diagnosis'].dropna().astype(str).unique().tolist()
    if not diagnosis_pool:
        return {"error": "No diagnosis data found in file."}

    # Boost context in the query
    prompt = f"This patient says: '{user_input.strip()}'. What is the most likely medical condition?"

    # Clean and embed
    def clean(text):
        return re.sub(r'[^a-zA-Z0-9\s]', '', text).lower().strip()

    cleaned_diagnoses = [clean(d) for d in diagnosis_pool]
    prompt_embedding = embedder.encode([clean(prompt)])[0]
    diagnosis_embeddings = embedder.encode(cleaned_diagnoses)

    similarities = cosine_similarity([prompt_embedding], diagnosis_embeddings)[0]
    best_idx = int(np.argmax(similarities))
    best_match = diagnosis_pool[best_idx]
    confidence = round(float(similarities[best_idx]) * 100, 2)

    return {
        "possible_condition": best_match,
        "confidence": f"{confidence}%"
    }



# === Perform Clustering on Text Columns with Descriptive Titles ===
def perform_clustering(column_name, num_clusters=5):
    if df.empty:
        return {"error": "Excel data not loaded."}
    if column_name not in df.columns:
        return {"error": f"Column '{column_name}' not found in the data."}

    texts = df[column_name].dropna().astype(str).tolist()
    if len(texts) < num_clusters:
        return {"error": f"Not enough entries in '{column_name}' to form {num_clusters} clusters."}

    embeddings = embedder.encode(texts)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)

    clusters = [[] for _ in range(num_clusters)]
    for text, label in zip(texts, labels):
        clusters[label].append(text)

    cluster_titles = {}
    for i in range(num_clusters):
        terms = [t.lower() for t in clusters[i]]
        words = [w for t in terms for w in re.findall(r'\b\w+\b', t)]
        common_words = pd.Series(words).value_counts().head(3).index.tolist()
        title = "Cluster {}: {}".format(i+1, ", ".join(common_words).title() if common_words else "General")
        cluster_titles[title] = clusters[i]

    return cluster_titles

# === Ask Route ===
@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    user_input = data.get("question", "").strip()
    print("[INFO] Question received:", user_input)

    result = diagnose_condition(user_input)
    return jsonify({"question": user_input, "response": result})

# === Cluster Route ===
@app.route("/cluster", methods=["POST"])
def cluster():
    data = request.get_json()
    column = data.get("column", "Symptoms")
    k = int(data.get("k", 5))

    result = perform_clustering(column_name=column, num_clusters=k)
    return jsonify({"column": column, "clusters": result})

if __name__ == "__main__":
    app.run(debug=True)