
import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://127.0.0.1:5000")
client = MlflowClient()

experiment = client.get_experiment_by_name("breast-cancer-optuna")

if experiment is None:
    print("Eksperimen tidak ditemukan.")
else:
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.mlflow.runName = 'trial-9'",
        max_results=5,
    )

    if not runs:
        print("Run trial-9 tidak ditemukan.")
    else:
        run = runs[0]
        print("Run ID:", run.info.run_id)
        print("Status:", run.info.status)

        artifacts = client.list_artifacts(run.info.run_id)

        print("\nDaftar artefak:")
        for artifact in artifacts:
            print(artifact.path, "| directory:", artifact.is_dir)
