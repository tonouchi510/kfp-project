from absl import app
from absl import flags
from google.cloud import aiplatform, bigquery 
from typing import List


FLAGS = flags.FLAGS
flags.DEFINE_string(
    "project_id", "",
    "id of gcp project.")

flags.DEFINE_string(
    "job_id", "",
    "id of pipeline job.")

flags.DEFINE_string(
    "featurestore_id", "",
    "id of featurestore.")

flags.DEFINE_string(
    "entity_type_id", "",
    "id of entity_type.")

flags.DEFINE_string(
    "entity_type_description", "",
    "description of entity_type.")

flags.DEFINE_string(
    "bucket_name", "",
    "GCS bucket name.")

flags.DEFINE_string(
    "location", "us-central1",
    "GCP location.")

flags.DEFINE_string(
    "bq_dataset_id", "",
    "bigquery dataset id.")

flags.DEFINE_string(
    "api_endpoint", "us-central1-aiplatform.googleapis.com",
    "aiplatform api endpoint.")  # 本当はsecretsとかで渡す方が良いかも


def create_entity_type(
    client,
    parent: str,
    entity_type_id: str,
    description: str = "sample entity type",
    timeout: int = 300,
):
    create_entity_type_request = aiplatform.gapic.CreateEntityTypeRequest(
        parent=parent,
        entity_type_id=entity_type_id,
        entity_type=aiplatform.gapic.EntityType(description=description),
    )
    lro_response = client.create_entity_type(request=create_entity_type_request)
    print("Long running operation:", lro_response.operation.name)
    create_entity_type_response = lro_response.result(timeout=timeout)
    print("create_entity_type_response:", create_entity_type_response)


def batch_create_features(
    client,
    parent: str,
    schemas: List,
    timeout: int = 300,
):
    """ FeatureStoreの特徴量定義をバッチで作成する.

    Args:
        client ([type]): FeatureStoreのService Client.
        parent (str): 親となるentity_typeへのパス.
        column_names (List): csvファイルから取り出したカラム配列
        column_types (List): 要素例: aiplatform.gapic.Feature.ValueType.INT64
        timeout (int, optional): リクエストタイムアウト. Defaults to 300.
    """
    requests = []
    for i in range(len(column_names)):
        feature = aiplatform.gapic.Feature(value_type=column_types[i])
        feature_request = aiplatform.gapic.CreateFeatureRequest(
            feature=feature, feature_id=column_names[i]
        )
        requests.append(feature_request)

    batch_create_features_request = aiplatform.gapic.BatchCreateFeaturesRequest(
        parent=parent, requests=requests
    )
    lro_response = client.batch_create_features(request=batch_create_features_request)
    print("Long running operation:", lro_response.operation.name)
    batch_create_features_response = lro_response.result(timeout=timeout)
    print("batch_create_features_response:", batch_create_features_response)


def get_schema(dataset_id: str, table_name: str):
    client = bigquery.Client()
    dataset_ref = bigquery.DatasetReference(client.project, dataset_id)
    table_ref = dataset_ref.table(table_name)
    table = client.get_table(table_ref)
    schemas = table.schema
    return schemas


def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    #client_options = {"api_endpoint": FLAGS.api_endpoint}
    #client = aiplatform.gapic.FeaturestoreServiceClient(client_options=client_options)
    schemas = get_schema(FLAGS.bq_dataset_id, FLAGS.table_name)
    print(schemas)

    """
    create_entity_type(
        client=client,
        parent=f"projects/{FLAGS.project_id}/locations/{FLAGS.location}/featurestores/{FLAGS.featurestore_id}",
        entity_type_id=FLAGS.entity_type_id,
        description=FLAGS.entity_type_description
    )

    entity_type = \
        f"projects/{FLAGS.project_id}/locations/{FLAGS.location} \
            /featurestores/{FLAGS.featurestore_id} \
            /entityTypes/{FLAGS.entity_type_id}"

    batch_create_features(
        client=client,
        parent=entity_type,
    )
    """


if __name__ == "__main__":
    app.run(main)
