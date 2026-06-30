import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

class ModelTrainer:
    def __init__(self, data: pd.DataFrame, target_column: str):
        self.X = data.drop(columns=[target_column])
        self.y = data[target_column]

    def train_and_evaluate(self, artifact_path: str):
        # Stratify to preserve target class balance during split
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.20, random_state=42, stratify=self.y
        )
        
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
        }
        
        best_model = None
        best_accuracy = 0.0
        
        for name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)
            print(f"[LOG] {name} Accuracy: {acc:.2%}")
            
            if acc > best_accuracy:
                best_accuracy = acc
                best_model = model
                
        print(f"[SUCCESS] Winner Selected: {best_model.__class__.__name__}")
        
        # Package feature names with model to verify future input shapes
        payload = {
            'model': best_model,
            'training_features': list(self.X.columns)
        }
        
        with open(artifact_path, 'wb') as f:
            pickle.dump(payload, f)