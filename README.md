🚀 SHL GenAI Assessment Recommender
Context-Aware Assessment Matching System

A production-style Generative AI powered recommendation engine that intelligently maps hiring requirements to the most relevant SHL assessments using semantic understanding instead of keyword matching.

🎯 Problem Statement

Recruiters often struggle to identify the right assessments for a role due to:

Large assessment catalogs

Overlapping skill tags

Manual keyword-based filtering

Lack of contextual understanding

This project solves that by introducing semantic similarity search powered by language embeddings.

🧠 System Overview

The system transforms both job descriptions and assessment metadata into vector representations and performs similarity-based retrieval to generate highly relevant recommendations.

It is built as a complete end-to-end AI application, including:

Data extraction & cleaning

Embedding generation

Similarity indexing

Model evaluation

Web-based interface

⚙️ How It Works
1️⃣ Text Embedding

All assessment descriptions are encoded into dense vector representations using a pre-trained Sentence-Transformer model.

2️⃣ Vector Similarity Search

Incoming hiring queries are embedded and matched against stored vectors using cosine similarity via a Nearest Neighbors index.

3️⃣ Lightweight Re-ranking

Domain-aware boosting improves precision by refining skill relevance signals.

📊 Evaluation Strategy

The system is validated using labeled training queries.

Metric Used:
Recall@10

This ensures that the correct assessment appears within the top 10 recommendations for the majority of evaluation queries.

🖥️ User Interface

A minimal, interactive browser interface enables:

Entering job descriptions

Selecting number of recommendations

Viewing ranked results with:

Assessment Name

Test Type

Skill Tags

Similarity Score

The UI is lightweight and designed for fast testing and demonstration.

🗂️ Project Architecture
SHL-GenAI-Assessment-Recommender/
│
├── app.py                     # Flask server & API endpoint
├── build_index.py             # Embedding + NN index builder
├── clean_catalog.py           # Data cleaning pipeline
├── retrieve_test.py           # CLI testing utility
│
├── data/
│   ├── catalog_clean.csv
│   ├── embeddings.npy
│   ├── metadata.csv
│   └── nn_model.joblib
│
├── templates/
│   └── index.html
│
├── static/
│   ├── script.js
│   └── styles.css
│
└── README.md
🛠 Technology Stack

Python

Flask

Sentence-Transformers

scikit-learn

Pandas / NumPy

HTML / CSS / JavaScript

✨ Key Strengths

✔ Context-aware semantic matching
✔ Modular architecture
✔ Fast inference using pre-built index
✔ Easy catalog extensibility
✔ Clean separation between backend and UI

📌 Practical Applications

This system can support:

Talent Acquisition Teams

Recruitment Agencies

HR Technology Platforms

Assessment Providers

by significantly reducing manual effort in test selection.

🔮 Future Enhancements

Hybrid keyword + semantic scoring

Feedback-based learning loop

FAISS indexing for large-scale catalogs

REST API deployment for enterprise integration

👨‍💻 Developed By

Mohammad Khadar Fhareed
SHL Research Engineer Assessment Submission
