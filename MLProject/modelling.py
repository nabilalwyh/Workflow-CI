import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment("Credit Scoring")

# Data preparation
X_train = pd.read_csv("dataset_preprocessing/X_train.csv")
y_train = pd.read_csv("dataset_preprocessing/y_train.csv").values.ravel()
X_test = pd.read_csv("dataset_preprocessing/X_test.csv")
y_test = pd.read_csv("dataset_preprocessing/y_test.csv").values.ravel()
input_example = X_train[0:5]

with mlflow.start_run():
    # Autolog otomatis akan simpan parameter, metrics, dan model
    mlflow.autolog()

    # Train model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Simpan model (opsional, autolog sudah simpan)
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        input_example=input_example
    )

    # Evaluasi model
    y_pred = model.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
