import os
from absl import app
from absl import flags
from google.cloud import aiplatform, storage
from typing import List


FLAGS = flags.FLAGS
flags.DEFINE_string(
    "project_name", "",
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
    "source_blob_name", "datasets/auto-mpg.csv",
    "dataset file path (csv).")

flags.DEFINE_string(
    "api_endpoint", "us-central1-aiplatform.googleapis.com",
    "aiplatform api endpoint.")  # 本当はsecretsとかで渡す方が良いかも


def download_blob(
    bucket_name: str,
    source_blob_name: str
) -> str:
    """Downloads a blob from the bucket if not exist."""
    # bucket_name = "your-bucket-name"
    # source_blob_name = "storage-object-name"

    destination_file_name = os.path.basename(source_blob_name)
    if os.path.exists(destination_file_name):
        print(f"Already exist: {destination_file_name}")
    else:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)

        blob = bucket.blob(source_blob_name)
        blob.download_to_filename(destination_file_name)

    return destination_file_name


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
    column_names: List,
    column_types: List,
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


def import_feature_values(
    client,
    csv_gcs_uri: str,
    column_names: List,
    entity_type: str,
    entity_id_field: str,
    feature_time_field: str,
    worker_count: int = 2,
):
    feature_specs = map(
        lambda column_name: aiplatform.gapic.ImportFeatureValuesRequest.FeatureSpec(id=column_name), column_names)

    import_feature_values_request = aiplatform.gapic.ImportFeatureValuesRequest(
        entity_type=entity_type,
        bigquery_source=aiplatform.gapic.BigQuerySource(input_uri="bq://furyu-nbiz.furyu_ml.auto-mpg"),
        feature_specs=feature_specs,
        entity_id_field=entity_id_field,
        worker_count=worker_count,
    )

    lro_response = client.import_feature_values(request=import_feature_values_request)
    print("Long running operation:", lro_response.operation.name)
    import_feature_values_response = lro_response.result(timeout=2000)
    print("import_feature_values_response:", import_feature_values_response)


def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    client_options = {"api_endpoint": FLAGS.api_endpoint}
    client = aiplatform.gapic.FeaturestoreServiceClient(client_options=client_options)

    create_entity_type(
        client=client,
        parent=f"projects/{FLAGS.project_id}/locations/{FLAGS.location}/featurestores/{FLAGS.featurestore_id}",
        entity_type_id=FLAGS.entity_type_id,
        description=FLAGS.entity_type_description
    )

    destination_file_name = download_blob(
        FLAGS.bucket_name,
        FLAGS.source_blob_name
    )

    column_names = []
    with open(destination_file_name) as f:
        column_names = f.readlines().strip("\t")
    column_types = []  # 何らか抽出する処理が必要

    entity_type = \
        f"projects/{FLAGS.project_id}/locations/{FLAGS.location} \
            /featurestores/{FLAGS.featurestore_id} \
            /entityTypes/{FLAGS.entity_type_id}"

    batch_create_features(
        client=client,
        parent=entity_type,
        column_names=column_names,
        column_types=column_types
    )

    import_feature_values(
        client=client,
        csv_gcs_uri=f"gs://{FLAGS.bucket_name}/{FLAGS.source_blob_name}",
        column_names=column_names,
        entity_type=entity_type,
        entity_id_field=,
        feature_time_field=
    )


if __name__ == "__main__":
    app.run(main)
