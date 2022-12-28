import kfp
from kfp import dsl
from kfp import gcp

# Initialize component store
component_store = kfp.components.ComponentStore(
    local_search_paths=["pipelines/hello-world-pipeline2", "components"])

# Create component factories
hello_op = component_store.load_component("hello")


# Define pipeline
@dsl.pipeline(
    name="hello-world-pipeline2",
    description="aaa"
)
def pipeline(
    pipeline_name: str = "",
    job_id: str = "xxxx",
    message: str = "hello world",
):
    hello_op(
        message=message,
    ).apply(gcp.use_preemptible_nodepool()) \
        .set_retry(num_retries=2)
