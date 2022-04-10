import kfp
from kfp import dsl
from kfp import gcp

# Initialize component store
component_store = kfp.components.ComponentStore(
    local_search_paths=["pipelines/head-pose-dataset-pipeline", "components"])

# Create component factories
pose_annotation_op = component_store.load_component("pose-annotation")
data_chunk_spliter_op = component_store.load_component("data-chunk-spliter")
slack_notification_op = component_store.load_component("slack-notification")


# Define pipeline
@dsl.pipeline(
    name="head-pose-dataset pipeline",
    description="Create tfrecord dataset for head-pose-pipeline."
)
def pipeline(
    pipeline_name: str = "head-pose-dataset-pipeline",
    bucket_name: str = "kfp-project",
    job_id: str = "{{JOB_ID}}",
    dataset: str = "",
    chunk_size: int = 1000,
    valid_ratio: float = 0.1,
):

    with dsl.ExitHandler(
        exit_op=slack_notification_op(
            pipeline_name=pipeline_name,
            job_id=job_id,
            message="Status: {{workflow.status}}"
        )
    ):
        split_task = data_chunk_spliter_op(
            pipeline=pipeline_name,
            bucket_name=bucket_name,
            job_id=job_id,
            dataset=dataset,
            chunk_size=chunk_size,
        ).apply(gcp.use_preemptible_nodepool()) \
            .set_retry(num_retries=2)

        with dsl.ParallelFor(split_task.output) as item:
            pose_annotation_op(
                pipeline=pipeline_name,
                bucket_name=bucket_name,
                job_id=job_id,
                valid_ratio=valid_ratio,
                chunk_file=item
            ).apply(gcp.use_preemptible_nodepool()) \
                .set_retry(num_retries=2)


if __name__ == "__main__":
    kfp.compiler.Compiler().compile(pipeline, "head-pose-dataset-pipeline.yaml")
