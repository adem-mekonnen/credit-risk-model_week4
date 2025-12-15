import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
import joblib
import os
import logging

logging.basicConfig(level=logging.INFO)

def train_model():
    input_path = "data/processed/train_data.csv"
    model_path = "src/api/best_model.pkl"
    
    try:
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"{input_path} not found. Run data_processing.py first.")
            
        logging.info("Loading training data...")
        df = pd.read_csv(input_path)
        
        X = df.drop(['AccountId', 'is_high_risk'], axis=1)
        y = df['is_high_risk']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        mlflow.set_experiment("Credit_Risk_Model")
        
        # Train Random Forest
        with mlflow.start_run(run_name="RandomForest"):
            logging.info("Training Random Forest...")
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X_train, y_train)
            
            # Metrics
            probs = rf.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, probs)
            acc = accuracy_score(y_test, rf.predict(X_test))
            
            logging.info(f"Random Forest AUC: {auc}")
            
            # Log
            mlflow.log_metric("auc", auc)
            mlflow.log_metric("accuracy", acc)
            mlflow.sklearn.log_model(rf, "model")
            
            # Save Artifact
            joblib.dump(rf, model_path)
            logging.info(f"Model saved to {model_path}")
            
    except Exception as e:
        logging.error(f"Training failed: {e}")

if __name__ == "__main__":
    train_model()