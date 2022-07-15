import kfp
from kfp import dsl
from kfp import gcp

# Initialize component store
component_store = kfp.components.ComponentStore(
    local_search_paths=["pipelines/simple-training-pipeline", "components"])

# Create component factories
train_op = component_store.load_component("training")
tensorboard_op = component_store.load_component("tb-observer")
slack_notification_op = component_store.load_component("slack-notification")


# Define pipeline
@dsl.pipeline(
    name="simple-training-pipeline",
    description="training pipeline for image classification"
)
def pipeline(
    pipeline_name: str = "simple-training-pipeline",
    bucket_name: str = "kfp-project",
    job_id: str = "{{JOB_ID}}",
    model_type: str = "resnet",
    global_batch_size: int = 1024,
    epochs: int = 30,
    lr: float = 0.001,
    image_size: int = 64,
    num_classes: int = 100,
    dataset: str = "gs://kfp-project/datasets/mnist",
):
    with dsl.ExitHandler(
        exit_op=slack_notification_op(
            pipeline_name=pipeline_name,
            job_id=job_id,
            message="Status: {{workflow.status}}"
        )
    ):
        train_op(
            pipeline_name=pipeline_name,
            bucket_name=bucket_name,
            job_id=job_id,
            global_batch_size=global_batch_size,
            epochs=epochs,
            learning_rate=lr,
            dataset=dataset,
            model_type=model_type,
            image_size=image_size,
            num_classes=num_classes,
        ).set_display_name("training")\
            .apply(gcp.use_preemptible_nodepool())\
            .apply(gcp.use_tpu(
                tpu_cores=8,
                tpu_resource="v3",
                tf_version="2.8.0"))\
            .set_retry(num_retries=2)

        tensorboard_op(
            pipeline_name=pipeline_name,
            bucket_name=bucket_name,
            job_id=job_id,
            tblog_dir="training/logs"
        ).set_display_name("tboard")\
            .apply(gcp.use_preemptible_nodepool())


if __name__ == "__main__":
    kfp.compiler.Compiler().compile(pipeline, "simple-training-pipeline.yaml")
