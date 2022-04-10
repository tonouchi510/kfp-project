import kfp
from kfp.v2 import dsl
from kfp import gcp

PIPELINE_NAME = "hello-world-pipeline"

# Initialize component store
component_store = kfp.components.ComponentStore(
    local_search_paths=["pipelines/hello-world-pipeline", "components"])

# Create component factories
hello_op = component_store.load_component("hello")


# Define pipeline
@dsl.pipeline(
    name="hello-world-pipeline",
    description="Output messages"
)
def pipeline(
    job_id: str = "xxxx",
    message: str = "hello world",
):
    hello_op(
        message=message,
    ).apply(gcp.use_preemptible_nodepool()) \
        .set_retry(num_retries=2)


if __name__ == "__main__":
    kfp.v2.compiler.Compiler().compile(
        pipeline_func=pipeline, 
        package_path="hello-world-pipeline.json"
    )
