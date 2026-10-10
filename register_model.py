
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

# Konfigurasi MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("breast-cancer-optuna")

client = MlflowClient()
model_name = "breast-cancer-rf"

# Dataset dan pembagian data
data = load_breast_cancer()
X = data.data
y = data.target

X_tr, X_val, y_tr, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Parameter trial-9
params = {
    "n_estimators": 300,
    "max_depth": 14,
    "min_samples_split": 8,
    "random_state": 42,
    "n_jobs": -1
}

model = RandomForestClassifier(**params)

with mlflow.start_run(run_name="trial-9-registered") as run:
    model.fit(X_tr, y_tr)

    pred = model.predict(X_val)
    proba = model.predict_proba(X_val)[:, 1]

    mlflow.log_params(params)
    mlflow.log_metric("val_f1", f1_score(y_val, pred))
    mlflow.log_metric("val_auc", roc_auc_score(y_val, proba))
    mlflow.log_metric("val_accuracy", accuracy_score(y_val, pred))

    # Simpan model sebagai artefak
    mlflow.sklearn.log_model(
        sk_model=model,
        name="model",
        input_example=X_val[:5],
        skops_trusted_types=["sklearn.tree._tree.Tree"]
    )

    model_uri = f"runs:/{run.info.run_id}/model"
    print("Run ID:", run.info.run_id)
    print("Model URI:", model_uri)

# Daftarkan model ke Model Registry
registered = mlflow.register_model(
    model_uri=model_uri,
    name=model_name
)

print("Nama model:", registered.name)
print("Versi model:", registered.version)
print("Registrasi model berhasil!")
