import kfp
from kfp import dsl
from kfp import gcp

PIPELINE_NAME = "training-pipeline"

# Initialize component store
component_store = kfp.components.ComponentStore(
    local_search_paths=["pipelines/training-pipeline", "components"])

# Create component factories
train_op = component_store.load_component("training")
tensorboard_op = component_store.load_component("tb-observer")
slack_notification_op = component_store.load_component("slack-notification")


# Define pipeline
@dsl.pipeline(
    name="training pipeline",
    description="training pipeline for simple cnn"
)
def pipeline(
    bucket: str = "kfp-project",
    job_id: str = '{{JOB_ID}}',
    dataset: str = "",
    batch_size: int = 1024,
    epochs: int = 30,
    lr: float = 0.001,
    model_type: str = "vit",
    emb_dim: int = 512,
):
    with dsl.ExitHandler(
        exit_op=slack_notification_op(
            pipeline=PIPELINE_NAME,
            bucket=bucket,
            jobid=job_id,
            message='Status: {{workflow.status}}'
        )
    ):
        training_task = train_op(
            bucket=bucket,
            jobid=job_id,
            batch_size=batch_size,
            epochs=epochs,
            lr=lr,
            dataset=dataset,
            model_type=model_type,
            emb_dim=emb_dim,
        ).set_display_name('training')\
            .apply(gcp.use_preemptible_nodepool())\
            .apply(gcp.use_tpu(
                tpu_cores=8,
                tpu_resource="preemptible-v3",
                tf_version="2.6.0"))\
            .set_retry(num_retries=2)

        tensorboard_op(
            bucket=bucket,
            jobid=job_id,
            log_dir='training/logs'
        ).set_display_name('tboard')\
            .apply(gcp.use_preemptible_nodepool())


if __name__ == "__main__":
    kfp.v2.compiler.Compiler().compile(
        pipeline_func=pipeline, 
        package_path="training-pipeline.json"
    )
