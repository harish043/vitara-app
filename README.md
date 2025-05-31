#  Vitara – VIT-Aware RAG Assistant

 Vitara is a contextual chatbot that augments a Large Language Model (Gemini 1.5 Pro) with semantically relevant data scraped from the VIT website and related sources. It employs Retrieval-Augmented Generation (RAG) using ChromaDB for embedding storage and semantic search, and serves responses via a sleek Streamlit-based chat interface.
🔧 What This Project Does

  - Data Ingestion: Scrapes structured and semi-structured content from VIT web sources to build a custom knowledge corpus.

  - Semantic Embedding with ChromaDB: Converts text data into high-dimensional vectors and stores them in ChromaDB to support fast, context-aware retrieval.

  - Retrieval-Augmented Generation (RAG): Uses the ChromaDB-retrieved context to enrich prompts for Gemini 1.5 Pro, improving response quality and factual alignment.

  - Chat Interface: Presents a user-friendly Streamlit chat UI for intuitive interaction with the augmented LLM.

  - Containerized Deployment: The full stack—including scraper, embedding logic, backend, and frontend—is containerized using Docker for consistent and reproducible execution.

  - Cloud Deployment: The Docker image is hosted on Docker Hub and deployed on Google Cloud Run, enabling a serverless, scalable chatbot experience.

# 🚀 Quick Start
## 🔹 Option 1: Run via Docker (Recommended)

  This is the easiest way to run the full app locally without setting up the database or scraping manually.

  - docker pull smilepolicy/vitara-app
  - docker run -p 8501:8501 smilepolicy/vitara-app

  Then open your browser and go to http://localhost:8501
## 🔹 Option 2: Clone and Run (Advanced)(NOT RECOMMENDED!!)

  - Note: This repository does not include the data scraping logic or vector database files. Use this only if you intend to build your own data pipeline.

  - git clone https://github.com/harish043/vitara-app.git
  - cd vitara-app
  - pip install -r requirements.txt
  - streamlit run app.py

  # 🛠️ Tech Stack

  - LLM: Gemini 1.5 Pro (via API)

  - Vector Store: ChromaDB

  - Embedding Model: OpenAI / Sentence Transformers (as configured)

  - Frontend: Streamlit

  - Containerization: Docker

  - Cloud Deployment: Google Cloud Run

  - Image Registry: Docker Hub – smilepolicy/vitara-app


🔒 Notes

  The ChromaDB database and source data are not included in this repository.

    To use this project locally without rebuilding the pipeline, pull and run the Docker container from Docker Hub.
