
import mlflow
import mlflow.sklearn

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

# 1. Hubungkan ke MLflow Tracking Server
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("breast-cancer-autolog")

# 2. Aktifkan pencatatan otomatis
mlflow.sklearn.autolog(log_models=True)

# 3. Muat dataset
X, y = load_breast_cancer(return_X_y=True)

# 4. Bagi data training dan validasi
X_tr, X_val, y_tr, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 5. Tentukan parameter model
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    random_state=42
)

# 6. Latih model dalam sebuah run MLflow
with mlflow.start_run(run_name="rf-autolog"):

    model.fit(X_tr, y_tr)

    # Hitung metrik validasi secara manual
    pred = model.predict(X_val)
    proba = model.predict_proba(X_val)[:, 1]

    f1 = f1_score(y_val, pred)
    auc = roc_auc_score(y_val, proba)
    accuracy = accuracy_score(y_val, pred)

    # Catat metrik validasi dengan nama khusus
    mlflow.log_metric("val_f1", f1)
    mlflow.log_metric("val_auc", auc)
    mlflow.log_metric("val_accuracy", accuracy)

    print("Autologging berhasil dijalankan!")
    print(f"Validation F1 Score : {f1:.4f}")
    print(f"Validation ROC-AUC  : {auc:.4f}")
    print(f"Validation Accuracy : {accuracy:.4f}")
