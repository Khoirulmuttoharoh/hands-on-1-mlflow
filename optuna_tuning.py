
import mlflow
import optuna

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

# 1. Hubungkan ke MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("breast-cancer-optuna")

# 2. Siapkan dataset
X, y = load_breast_cancer(return_X_y=True)

X_tr, X_val, y_tr, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 3. Fungsi untuk menguji satu kombinasi parameter
def objective(trial):

    # Optuna memilih kombinasi parameter
    params = {
        "n_estimators": trial.suggest_int(
            "n_estimators", 100, 300, step=100
        ),
        "max_depth": trial.suggest_int(
            "max_depth", 3, 15
        ),
        "min_samples_split": trial.suggest_int(
            "min_samples_split", 2, 10
        ),
        "random_state": 42,
        "n_jobs": -1
    }

    # Setiap percobaan menjadi nested run
    with mlflow.start_run(
        run_name=f"trial-{trial.number}",
        nested=True
    ):
        mlflow.log_params(params)

        model = RandomForestClassifier(**params)
        model.fit(X_tr, y_tr)

        pred = model.predict(X_val)
        proba = model.predict_proba(X_val)[:, 1]

        f1 = f1_score(y_val, pred)
        auc = roc_auc_score(y_val, proba)
        accuracy = accuracy_score(y_val, pred)

        mlflow.log_metric("val_f1", f1)
        mlflow.log_metric("val_auc", auc)
        mlflow.log_metric("val_accuracy", accuracy)

        print(
            f"Trial {trial.number}: "
            f"F1={f1:.4f}, AUC={auc:.4f}, "
            f"Accuracy={accuracy:.4f}"
        )

    # Optuna berusaha memaksimalkan F1 Score
    return f1


# 4. Buat study dan catat parent run
with mlflow.start_run(run_name="optuna-parent"):

    study = optuna.create_study(direction="maximize")

    # Jalankan 10 percobaan
    study.optimize(objective, n_trials=10)

    # 5. Simpan hasil terbaik ke parent run
    mlflow.log_metric("best_val_f1", study.best_value)
    mlflow.log_params({
        f"best_{key}": value
        for key, value in study.best_params.items()
    })

    print("\n=== HASIL TERBAIK ===")
    print(f"Best F1 Score: {study.best_value:.4f}")
    print(f"Best Parameters: {study.best_params}")
