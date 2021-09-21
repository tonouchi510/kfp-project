import kfp
from kfp import dsl
from kfp import gcp

PIPELINE_NAME = "hello-world-pipeline"

# Initialize component store
component_store = kfp.components.ComponentStore(
    local_search_paths=["pipelines/hello-world-pipeline", "components"])

# Create component factories
parallelizer_op = component_store.load_component("parallelizer")
hello_op = component_store.load_component("hello")
slack_notification_op = component_store.load_component("slack-notification")


# Define pipeline
@dsl.pipeline(
    name="hello world pipeline",
    description="Output messages"
)
def pipeline(
    bucket: str = "xxxx",
    job_id: str = "xxxx",
    message_file: str = "gs://",
):

    with dsl.ExitHandler(
        exit_op=slack_notification_op(
            pipeline_name=PIPELINE_NAME,
            job_id=job_id,
            message="Status: {{workflow.status}}"
        )
    ):
        parallelizer_task = parallelizer_op(
            bucket=bucket,
            job_id=job_id,
            message_file=message_file,
        ).apply(gcp.use_preemptible_nodepool()) \
            .set_retry(num_retries=2)

        with dsl.ParallelFor(parallelizer_task.output) as item:
            hello_task = hello_op(
                message=item,
            ).apply(gcp.use_preemptible_nodepool()) \
                .set_retry(num_retries=2)


if __name__ == "__main__":
    kfp.compiler.Compiler().compile(pipeline, "hello-world-pipeline.yaml")
