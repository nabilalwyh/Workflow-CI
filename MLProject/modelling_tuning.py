import os
import json
import pandas as pd
import numpy as np
import mlflow
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from mlflow.models.signature import infer_signature
from mlflow.sklearn import log_model
import warnings
import tempfile

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)
    
    # === Load dataset yang benar ===
    X_train = pd.read_csv("dataset_preprocessing/X_train.csv")
    y_train = pd.read_csv("dataset_preprocessing/y_train.csv").squeeze()
    X_test = pd.read_csv("dataset_preprocessing/X_test.csv")
    y_test = pd.read_csv("dataset_preprocessing/y_test.csv").squeeze()

    example_input = X_train.head(5)

    # === Jalankan experiment MLflow ===
    with mlflow.start_run(run_name="RF_RandomizedSearch"):
        # Space hyperparameter
        param_dist = {
            "n_estimators": [100, 200, 300, 400],
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 4, 6],
            "min_samples_leaf": [1, 2, 4]
        }

        search = RandomizedSearchCV(
            estimator=RandomForestClassifier(random_state=42),
            param_distributions=param_dist,
            n_iter=8,
            scoring="f1",
            cv=3,
            n_jobs=-1,
            random_state=42
        )
        search.fit(X_train, y_train)

        best_model = search.best_estimator_
        best_params = search.best_params_

        # === Evaluasi ===
        preds = best_model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)

        # Logging manual
        mlflow.log_params(best_params)
        mlflow.log_metrics({
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1
        })

        # Log model dengan signature
        sig = infer_signature(X_test, preds)
        mlflow.sklearn.log_model(best_model, "rf_best_model",
                                input_example=example_input,
                                signature=sig)

        # === Artefak tambahan ===
        with tempfile.TemporaryDirectory() as tmp:
            # Confusion Matrix
            cm = confusion_matrix(y_test, preds)
            plt.figure(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
            plt.title("Confusion Matrix")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            cm_path = f"{tmp}/confusion_matrix.png"
            plt.savefig(cm_path)

            # Classification report
            report_path = f"{tmp}/classification_report.txt"
            with open(report_path, "w") as f:
                f.write(classification_report(y_test, preds))

            # Metrics JSON
            metrics_path = f"{tmp}/metrics.json"
            with open(metrics_path, "w") as f:
                json.dump({"accuracy": acc, "precision": prec,
                        "recall": rec, "f1_score": f1}, f, indent=2)

            # Model summary HTML
            html_path = f"{tmp}/model_summary.html"
            html_content = f"""
            <html>
            <head><title>Model Summary</title></head>
            <body>
            <h2>Best RandomForest Estimator</h2>
            <pre>{best_model}</pre>
            <h3>Best Parameters</h3>
            <ul>
            {''.join([f"<li>{k}: {v}</li>" for k, v in best_params.items()])}
            </ul>
            <p><b>Cross-Val F1 Score:</b> {search.best_score_:.4f}</p>
            </body></html>
            """
            with open(html_path, "w") as f:
                f.write(html_content)

            # Log semua artefak
            mlflow.log_artifacts(tmp)

        # Output ke terminal
        print("\n=== Best Params ===")
        print(best_params)
        print("\n=== Classification Report ===")
        print(classification_report(y_test, preds))
