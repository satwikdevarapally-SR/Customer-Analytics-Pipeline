import os
from src.ingestion import DataIngestion
from src.preprocessing import DataPreprocessor
from src.training import ModelTrainer

def run_pipeline():
    RAW_DATA_PATH = "data/raw_customer_data.csv"
    ARTIFACT_PATH = "artifacts/optimal_model.pkl"
    
    os.makedirs("artifacts", exist_ok=True)
    
    # Extract
    ingestion = DataIngestion(raw_data_path=RAW_DATA_PATH)
    raw_df = ingestion.load_data()
    
    # Transform
    clean_df = DataPreprocessor.clean_features(raw_df)
    
    # Load / Train
    trainer = ModelTrainer(data=clean_df, target_column="ChurnStatus")
    trainer.train_and_evaluate(artifact_path=ARTIFACT_PATH)

if __name__ == "__main__":
    run_pipeline()