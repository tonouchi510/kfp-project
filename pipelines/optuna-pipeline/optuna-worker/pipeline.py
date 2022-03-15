import os
import time
import kfp
import yaml
from typing import Tuple
import pandas as pd
from pathlib import Path
from google.cloud import storage


def run_pipeline(
    host_url: str,
    job_id: str,
    pipeline_name: str,
    trial_number: int
):
    """Run the training-pipeline for optuna.

    Args:
        host_url (str): kubeflow pipelinesのhost url.
        job_id (str): optunaパイプライン側のjob_id.
        pipeline_name (str): optuna最適化対象のトレーニングパイプライン名.
        trial_id (str): optuna trialごとの固有id.

    Returns:
        [type]: [description]
    """
    client = kfp.Client(host=host_url)
    # pipeline_nameのデフォ（最新version）のpipeline_idを取得
    pipeline_id = client.get_pipeline_id(name=pipeline_name)

    experiment_name = "optuna-job"
    try:
        experiment = client.get_experiment(experiment_name=experiment_name)
    except Exception as e:
        print(e)
        print(f"creating {experiment_name}.")
        experiment = client.create_experiment(name=experiment_name)

    job_name = f"{job_id}-{trial_number}"
    settings = read_settings(job_id, trial_number)
    client.run_pipeline(
        experiment.id,
        job_name=job_name,
        pipeline_id=pipeline_id,
        params=settings)


def read_settings(job_id: str, trial_number: int):
    """Read all the parameter values from the settings.yaml file.

    Args:
        job_id (str): optunaパイプライン側のjob_id
        trial_id (str): optuna trialごとの固有id.

    Returns:
        [type]: [description]
    """
    settings_file = f"{job_id}-{trial_number}.yaml"
    flat_settings = dict()
    setting_sections = yaml.safe_load(Path(settings_file).read_text())
    for sections in setting_sections:
        setting_sections[sections]["job_id"] = f"{job_id}-{trial_number}"
        flat_settings.update(setting_sections[sections])
    return flat_settings


def get_pipeline_result(
    job_id: str,
    bucket_name: str,
    pipeline_name: str,
    trial_number: int
) -> Tuple[float, int]:
    trial_history = f"artifacts/{pipeline_name}/{job_id}-{trial_number}/training/history.csv"
    destination_file_name = os.path.basename(trial_history)
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    blobs = bucket.list_blobs(prefix=trial_history)
    exist = None
    while not exist:
        time.sleep(360)
        blobs = bucket.list_blobs(prefix=trial_history)
        for b in blobs:
            exist = b

    blob = bucket.blob(trial_history)
    blob.download_to_filename(destination_file_name)

    df = pd.read_csv(destination_file_name, index_col=0)
    val_loss = min(list(df["val_loss"].astype("float")))
    return val_loss, df.index[-1] + 1
