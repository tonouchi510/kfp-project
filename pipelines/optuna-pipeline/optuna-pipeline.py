import kfp
from kfp import dsl
import os

# Initialize component store
component_store = kfp.components.ComponentStore(
    local_search_paths=["pipelines/optuna-pipeline", "components"])

# Create component factories
optuna_op = component_store.load_component("optuna-worker")
slack_notification_op = component_store.load_component("slack-notification")

sidecar = kfp.dsl.Sidecar(
    name="cloudsqlproxy",
    image="gcr.io/cloudsql-docker/gce-proxy:1.14",
    command=[
        "/cloud_sql_proxy",
        f"-instances={os.environ.get('GCP_PROJECT')}:{os.environ.get('GCP_REGION')}:{os.environ.get('DB_NAME')}=tcp:3306",
    ],
)

# Define pipeline
@dsl.pipeline(
    name="optuna pipeline",
    description="optuna pipeline"
)
def pipeline(
    pipeline_name: str = "optuna-pipeline",
    bucket_name: str = "",
    job_id: str = "{{JOB_ID}}",
    n_trials: int = 100,
    n_jobs: int = 5,
    training_pipeline_name: str = "head-pose-pipeline",
    dataset: str = "",
    epochs: int = 5
):
    with dsl.ExitHandler(
        exit_op=slack_notification_op(
            pipeline_name=pipeline_name,
            job_id=job_id,
            message="Status: {{workflow.status}}"
        ).add_node_selector_constraint("cloud.google.com/gke-nodepool", "main-pool")
    ):
        optuna_op(
            pipeline_name=pipeline_name,
            bucket_name=bucket_name,
            job_id=job_id,
            n_trials=n_trials,
            n_jobs=n_jobs,
            training_pipeline_name=training_pipeline_name,
            dataset=dataset,
            epochs=epochs
        ).set_display_name("optuna-worker")\
            .add_node_selector_constraint("cloud.google.com/gke-nodepool", "cpu-pool")\
            .add_sidecar(sidecar)\
            .set_retry(num_retries=2)


if __name__ == "__main__":
    kfp.compiler.Compiler().compile(pipeline, "optuna-pipeline.yaml")
