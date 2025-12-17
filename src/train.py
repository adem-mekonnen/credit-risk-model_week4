import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import joblib
import os
import logging

logging.basicConfig(level=logging.INFO)

def evaluate_metrics(y_true, y_pred, y_prob):
    """Calculates all required metrics."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob)
    }

def train_model():
    data_path = "data/processed/train_data.csv"
    model_path = "src/api/best_model.pkl"
    
    if not os.path.exists(data_path):
        logging.error(f"{data_path} not found. Run data_processing.py first.")
        return

    df = pd.read_csv(data_path)
    X = df.drop(['AccountId', 'is_high_risk'], axis=1)
    y = df['is_high_risk']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    mlflow.set_experiment("Credit_Risk_Xente_Final")
    
    models_to_tune = {
        "LogisticRegression": (LogisticRegression(max_iter=1000, random_state=42), 
                               {'C': [0.01, 0.1, 1]}),
        "RandomForest": (RandomForestClassifier(random_state=42), 
                         {'n_estimators': [50, 100], 'max_depth': [5, 10]})
    }
    
    best_auc = 0
    best_estimator = None
    
    for name, (model, params) in models_to_tune.items():
        with mlflow.start_run(run_name=name):
            # Task 5: Hyperparameter Tuning (Grid Search)
            logging.info(f"Starting Grid Search for {name}...")
            clf = GridSearchCV(model, params, cv=3, scoring='roc_auc', n_jobs=-1)
            clf.fit(X_train, y_train)
            
            # Use best estimator for final evaluation
            final_model = clf.best_estimator_
            y_pred = final_model.predict(X_test)
            y_prob = final_model.predict_proba(X_test)[:, 1]
            
            metrics = evaluate_metrics(y_test, y_pred, y_prob)
            
            # Log to MLflow
            mlflow.log_params(clf.best_params_)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(final_model, name, registered_model_name=f"{name}_Credit_Risk")

            # Log Confusion Matrix as Artifact
            cm = confusion_matrix(y_test, y_pred)
            pd.DataFrame(cm).to_csv("confusion_matrix.csv", index=False)
            mlflow.log_artifact("confusion_matrix.csv")
            os.remove("confusion_matrix.csv")
            
            logging.info(f"{name} - Metrics: {metrics}")
            
            if metrics['roc_auc'] > best_auc:
                best_auc = metrics['roc_auc']
                best_estimator = final_model
                
    # Save Best Model locally for FastAPI to load
    joblib.dump(best_estimator, model_path)
    logging.info("Best model saved to src/api/best_model.pkl")

if __name__ == "__main__":
    train_model()