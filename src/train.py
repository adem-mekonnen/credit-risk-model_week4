import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
import joblib
import os

def train():
    data_path = "data/processed/train_data.csv"
    model_path = "src/api/best_model.pkl"
    
    if not os.path.exists(data_path):
        print("Processed data not found. Run data_processing.py first.")
        return

    df = pd.read_csv(data_path)
    
    # Split
    X = df.drop(['AccountId', 'is_high_risk'], axis=1)
    y = df['is_high_risk']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    mlflow.set_experiment("Credit_Risk_Xente")
    
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    best_auc = 0
    best_model = None
    
    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            model.fit(X_train, y_train)
            
            # Predict
            probs = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, probs)
            acc = accuracy_score(y_test, model.predict(X_test))
            
            # Log
            mlflow.log_metric("auc", auc)
            mlflow.log_metric("accuracy", acc)
            mlflow.sklearn.log_model(model, name)
            
            print(f"{name} AUC: {auc:.4f}")
            
            if auc > best_auc:
                best_auc = auc
                best_model = model
    
    # Save Best Model
    joblib.dump(best_model, model_path)
    print(f"Best model saved to {model_path} with AUC: {best_auc:.4f}")

if __name__ == "__main__":
    train()