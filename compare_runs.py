
import mlflow
from mlflow.tracking import MlflowClient

# Hubungkan ke server MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")

client = MlflowClient()

# Cari eksperimen Optuna
experiment = client.get_experiment_by_name(
    "breast-cancer-optuna"
)

if experiment is None:
    print("Eksperimen breast-cancer-optuna tidak ditemukan.")

else:
    # Ambil run yang berhasil dan urutkan berdasarkan F1 Score
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["metrics.val_f1 DESC"],
        max_results=20
    )

    print("\n=== PERBANDINGAN HASIL OPTUNA ===")

    if not runs:
        print("Belum ada run yang selesai.")
    else:
        for rank, run in enumerate(runs, start=1):
            print(f"\nPeringkat: {rank}")
            print(
                "Run name:",
                run.data.tags.get("mlflow.runName", "N/A")
            )
            print(
                "F1 Score:",
                run.data.metrics.get("val_f1", "N/A")
            )
            print(
                "ROC-AUC:",
                run.data.metrics.get("val_auc", "N/A")
            )
            print(
                "Accuracy:",
                run.data.metrics.get("val_accuracy", "N/A")
            )
            print("Parameters:", run.data.params)

        # Tampilkan trial terbaik
        best_run = runs[0]

        print("\n=== TRIAL TERBAIK ===")
        print("Run ID:", best_run.info.run_id)
        print(
            "Nama:",
            best_run.data.tags.get("mlflow.runName", "N/A")
        )
        print(
            "F1 Score terbaik:",
            best_run.data.metrics.get("val_f1", "N/A")
        )
        print("Parameter terbaik:", best_run.data.params)
