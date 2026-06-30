customer-analytics-pipeline/
├── artifacts/
│   └── optimal_model.pkl       # Saved binary payload (model + feature names)
├── data/
│   └── raw_customer_data.csv   # Source customer records
├── src/
│   ├── __init__.py
│   ├── ingestion.py            # Phase 1: Data extraction and structural validation
│   ├── preprocessing.py         # Phase 2: Sanitization, encoding, and imputation
│   └── training.py             # Phase 3: Stratified splitting and model competition
├── app/
│   └── server.py               # FastAPI inference backend
└── main.py                     # Pipeline master orchestrator
