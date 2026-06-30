# Production-Grade Customer Churn ML Pipeline

A modular, enterprise-ready machine learning and ETL pipeline built with Object-Oriented Programming (OOP) principles to predict customer subscription churn. 

Rather than utilizing a monolithic script or a messy Jupyter Notebook, this system is architected as a series of decoupled microservices—separating ingestion, engineering, and training logic to mirror real-world production data pipelines.

---

## 🏗️ System Architecture

The pipeline decouples infrastructure from core computing logic, allowing independent scaling of data transformations and model training.

```text
       [ raw_customer_data.csv ]
                   │
                   ▼
         ┌───────────────────┐
         │   ingestion.py    │  <-- Structural Verification & Extraction
         └─────────┬─────────┘
                   │
                   ▼
         ┌───────────────────┐
         │ preprocessing.py  │  <-- Vectorized Cleaning & Dummy Variable Alignment
         └─────────┬─────────┘
                   │
                   ▼
         ┌───────────────────┐
         │    training.py    │  <-- Stratified Arena / Model Evaluation
         └─────────┬─────────┘
                   │
                   ▼
       [ optimal_model.pkl ]    <-- Packaged Model + Feature Metadata Payload
                   │
                   ▼
         ┌───────────────────┐
         │     server.py     │  <-- FastAPI Inference Engine
         └───────────────────┘
