
import mlflow
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Muat model berdasarkan nama dan versi di Model Registry
model_uri = "models:/breast-cancer-rf/1"
model = mlflow.sklearn.load_model(model_uri)

# Siapkan data pengujian yang sama
data = load_breast_cancer()
X = data.data
y = data.target

X_tr, X_val, y_tr, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Prediksi menggunakan model yang sudah terdaftar
pred = model.predict(X_val)
proba = model.predict_proba(X_val)[:, 1]

print("=== HASIL PENGUJIAN MODEL TERDAFTAR ===")
print("Model URI:", model_uri)
print("Accuracy:", accuracy_score(y_val, pred))
print("F1-score:", f1_score(y_val, pred))
print("ROC-AUC:", roc_auc_score(y_val, proba))
print("Model berhasil dimuat dan digunakan untuk prediksi.")
