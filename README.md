# Zepto AI Platform

A multi-part data and AI project featuring a book data pipeline, Titanic dataset analytics, and a Zepto policy question-answering API.

## Project Overview

This repository contains three components:

1. **Book Data Pipeline** — scrapes book listings, cleans and transforms the data, stores it in SQLite, and demonstrates SQL and pandas analysis.
2. **Titanic Data Analytics** — profiles and cleans the Titanic dataset, explores patterns through visualizations, trains classification models, and predicts fare values.
3. **Zepto Policy Support Assistant** — a retrieval-augmented generation (RAG) API that answers questions using local Zepto policy documents.

## Repository Structure

```text
zepto-ai-platform/
├── analytics/
│   ├── analysis.py
│   ├── titanic.csv
│   └── models/
│       └── best_pipeline.joblib
├── data_pipeline/
│   ├── scraper.py
│   ├── database.py
│   ├── queries.py
│   ├── books.csv
│   └── books.db
├── support_assistant/
│   ├── docs/
│   │   ├── doc_01.txt
│   │   ├── doc_02.txt
│   │   ├── ...
│   │   └── doc_08.txt
│   ├── main.py
│   ├── app.py
│   ├── requirements.txt
│   ├── Dockerfile
│   └── README.md
├── .gitignore
└── README.md
