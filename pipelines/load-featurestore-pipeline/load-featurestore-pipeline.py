import kfp
from kfp import dsl
from kfp import gcp

PIPELINE_NAME = "load-featurestore-pipeline"

# Initialize component store
component_store = kfp.components.ComponentStore(
    local_search_paths=["pipelines/load-featurestore-pipeline", "components"])

# Create component factories
# load_op = component_store.load_component("load")
create_entity_op = component_store.load_component("create-entity")
slack_notification_op = component_store.load_component("slack-notification")


# Define pipeline
@dsl.pipeline(
    name="load-featurestore-pipeline",
    description="load features to featurestore"
)
def pipeline(
    bucket_name: str = "kfp-project",
    job_id: str = "{{JOB_ID}}",
    location: str = "us-central1",
    featurestore_id: str = "",
    entity_type_id: str = "",
    entity_type_description: str = "",
    source_blob_name: str = "",
    api_endpoint: str = "",
    bq_dataset_id: str = "",
    bq_table_name: str = ""
):
    with dsl.ExitHandler(
        exit_op=slack_notification_op(
            pipeline_name=PIPELINE_NAME,
            job_id=job_id
        )
    ):
        create_entity_op(
            pipeline_name=PIPELINE_NAME,
            job_id=job_id,
            bucket_name=bucket_name,
            location=location,
            featurestore_id=featurestore_id,
            entity_type_id=entity_type_id,
            entity_type_description=entity_type_description,
            source_blob_name=source_blob_name,
            api_endpoint=api_endpoint,
            bq_dataset_id=bq_dataset_id,
            bq_table_name=bq_table_name
        ).apply(gcp.use_preemptible_nodepool())\
            .set_retry(num_retries=2)


if __name__ == "__main__":
    kfp.v2.compiler.Compiler().compile(
        pipeline_func=pipeline, 
        package_path="load-featurestore-pipeline.json"
    )
