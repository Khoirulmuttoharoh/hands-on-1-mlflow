
import mlflow
import mlflow.sklearn

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

# Menghubungkan program ke server MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Membuat eksperimen
mlflow.set_experiment("breast-cancer-clf")

# Memuat dataset
X, y = load_breast_cancer(return_X_y=True)

# Membagi data training dan validasi
X_tr, X_val, y_tr, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Parameter model
params = {
    "n_estimators": 200,
    "max_depth": 8,
    "random_state": 42
}

# Memulai pencatatan eksperimen
with mlflow.start_run(run_name="rf-baseline"):

    mlflow.set_tags({
        "stage": "baseline",
        "author": "mahasiswa"
    })

    mlflow.log_params(params)

    # Melatih model
    model = RandomForestClassifier(**params)
    model.fit(X_tr, y_tr)

    # Prediksi
    pred = model.predict(X_val)
    proba = model.predict_proba(X_val)[:, 1]

    # Evaluasi model
    f1 = f1_score(y_val, pred)
    auc = roc_auc_score(y_val, proba)
    accuracy = accuracy_score(y_val, pred)

    # Mencatat metrik
    mlflow.log_metric("val_f1", f1)
    mlflow.log_metric("val_auc", auc)
    mlflow.log_metric("val_accuracy", accuracy)

    # Menyimpan model ke MLflow
    mlflow.sklearn.log_model(
    sk_model=model,
    name="model",
    input_example=X_val[:5],
    skops_trusted_types=["sklearn.tree._tree.Tree"]
    )

    print("Eksperimen berhasil!")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {auc:.4f}")
    print(f"Accuracy : {accuracy:.4f}")
