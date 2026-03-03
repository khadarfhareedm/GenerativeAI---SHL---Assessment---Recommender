**# 🚀 SHL – Generative AI Assessment Recommendation System

A semantic recommendation engine that intelligently maps hiring queries to the most relevant SHL assessments using contextual language understanding instead of simple keyword matching.

---

## 📌 Project Overview

This system helps recruiters and hiring teams quickly identify the most suitable SHL assessments for a given job description.

Instead of relying on keyword filtering, the system understands the **intent and context** of the query using modern NLP embeddings and similarity search.

---

## 🎯 What This Project Does

- Accepts a **job description / hiring query**
- Converts text into **dense semantic embeddings**
- Matches against SHL assessment descriptions
- Returns **Top-K ranked recommendations**
- Displays similarity scores for transparency

---

## 🧠 Core Approach

### 1️⃣ Semantic Representation
Assessment descriptions and user queries are converted into vector embeddings using a **Sentence-Transformer model**.

### 2️⃣ Similarity Search
Cosine similarity is used to retrieve the most relevant assessments via a **Nearest Neighbors index**.

### 3️⃣ Intelligent Re-Ranking
Lightweight domain-aware boosting improves skill-level relevance.

---

## 📊 Evaluation

The system is evaluated using labeled training queries.

**Metric Used:**  
`Recall@10`

The evaluation confirms that the correct assessment appears within the top 10 recommendations for most queries.

---

## 🖥️ User Interface Features

Users can:

- Enter job descriptions
- Select number of recommendations
- View ranked results including:
  - Assessment Name
  - Test Type
  - Skill Tags
  - Similarity Score

The interface is lightweight and optimized for fast testing and demonstration.

---

## 🛠 Technology Stack

- Python  
- Flask  
- Sentence-Transformers  
- scikit-learn  
- Pandas  
- NumPy  
- HTML / CSS / JavaScript  

---

## 🗂 Project Structure


SHL-GenAI-Assessment-Recommender/
│
├── app.py # Flask API & recommendation logic
├── build_index.py # Embedding + NN index builder
├── clean_catalog.py # Data cleaning pipeline
├── retrieve_test.py # CLI testing utility
│
├── data/
│ ├── catalog_clean.csv
│ ├── embeddings.npy
│ ├── metadata.csv
│ └── nn_model.joblib
│
├── templates/
│ └── index.html
│
├── static/
│ ├── script.js
│ └── styles.css
│
└── README.md


---

## ✨ Key Highlights

- Context-aware semantic matching  
- Modular and clean architecture  
- Fast inference using pre-built index  
- Easy to extend with new assessments  
- Separation of data, model, and UI layers  

---

## 📌 Practical Applications

This system can be used by:

- Recruiters  
- Talent Acquisition Teams  
- HR Technology Platforms  
- Assessment Vendors  

to efficiently shortlist relevant assessments for specific roles.

---

## 👨‍💻 Developed By

**Mohammad Khadar Fhareed**  
SHL Research Engineer – Assessment Submission**
