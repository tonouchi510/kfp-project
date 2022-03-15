import ast
import yaml
from absl import flags, app
from kfp_server_api.rest import ApiException
from logging import getLogger
from pipeline import run_pipeline, get_pipeline_result
import optuna
from optuna import Trial

logger = getLogger(__name__)
PROJECT_ID = "YOUR_PROJECT"
KFP_HOST = ""

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "pipeline_name", None,
    "Name of pipeline")

flags.DEFINE_string(
    "bucket_name", "",
    "GCS bucket name")

flags.DEFINE_string(
    "job_id", "test",
    "ID for job management.")

flags.DEFINE_integer(
    "n_trials", 50,
    "Max trial num for optuna.")

flags.DEFINE_integer(
    "n_jobs", 5,
    "Num of parallel for optuna.")

flags.DEFINE_string(
    "training_pipeline_name", "head-pose-pipeline",
    "Pipeline name for training.")

flags.DEFINE_string(
    "dataset", "",
    "Directory where dataset is stored.")

flags.DEFINE_integer(
    "epochs", 30,
    "Num of training epochs.")


def access_secret(
    secret_id: str,
    version_id: str = "latest"
):
    """
    Access the payload for the given secret version if one exists. The version
    can be a version number as a string (e.g. "5") or an alias (e.g. "latest").
    """
    from google.cloud import secretmanager

    client = secretmanager.SecretManagerServiceClient()

    name = f"projects/{PROJECT_ID}/secrets/{secret_id}/versions/{version_id}"
    response = client.access_secret_version(request={"name": name})

    payload = response.payload.data.decode("UTF-8")
    res = ast.literal_eval(payload)
    return res


def objective(trial: Trial) -> float:
    # 終了判定（たまにn_trialsで停止しないバグが生じるため）
    completed_trials = []
    for t in trial.study.trials:
        if t.state == optuna.structs.TrialState.COMPLETE or t.state == optuna.structs.TrialState.FAIL:
            completed_trials.append(1)

    if len(completed_trials) >= FLAGS.n_trials:
        logger.info(f"Number of completed trials: {len(completed_trials)}")
        trial.study.stop()
        return

    # trialごとのsettings.yamlの構成
    create_settings(trial)

    # トレーニングパイプラインの実行(optunaの1試行)
    run_pipeline(
        host_url=KFP_HOST,
        job_id=FLAGS.job_id,
        pipeline_name=FLAGS.training_pipeline_name,
        trial_number=trial.number
    )

    # 結果を受け取る
    result, last_epoch = get_pipeline_result(
        job_id=FLAGS.job_id,
        bucket_name=FLAGS.bucket,
        pipeline_name=FLAGS.training_pipeline_name,
        trial_number=trial.number
    )
    logger.info(
        f"Finished trial#{trial.number} result: {result}, epoch: {last_epoch}, params: {trial.params}")
    return result


def create_settings(trial):
    settings_file = f"{FLAGS.job_id}-{trial.number}.yaml"

    if FLAGS.training_pipeline_name == "head-pose-pipeline":
        image_size = trial.suggest_categorical(
            "image_size", [64, 96, 112])
        model_type = trial.suggest_categorical(
            "model_type", [0, 1, 2, 3, 4, 5, 6, 7])
        learning_rate = trial.suggest_categorical(
            "lr", [1e-5, 1e-4, 1e-3, 1e-2])

        with open(settings_file, "w") as fp:
            yaml.dump({
                "arguments": {
                    "pipeline_name": FLAGS.training_pipeline_name,
                    "bucket_name": FLAGS.bucket_name,
                    "model_type": model_type,
                    "global_batch_size": 1024,
                    "epochs": FLAGS.epochs,
                    "lr": learning_rate,
                    "image_size": image_size,
                    "dataset": FLAGS.dataset,
                }
            }, fp)
    else:
        raise ValueError(
            f"Invalid param error: training_pipeline={FLAGS.training_pipeline_name}.")


class StopWhenTrialKeepBeingPrunedCallback:
    def __init__(self, threshold: int):
        self.threshold = threshold
        self._consequtive_pruned_count = 0

    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
        if trial.state == optuna.trial.TrialState.PRUNED:
            self._consequtive_pruned_count += 1
        else:
            self._consequtive_pruned_count = 0

        if self._consequtive_pruned_count >= self.threshold:
            study.stop()


def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    KFP_HOST = access_secret(secret_id="kubeflow_host")["HOST"]
    res = access_secret(secret_id="optuna-db-secret")
    study_storage = f"mysql+pymysql://{res['USER']}:{res['PASSWORD']}@localhost/optuna"

    study = optuna.create_study(
        direction="minimize",
        study_name=FLAGS.job_id,
        storage=study_storage,
        load_if_exists=True,
    )

    study.optimize(
        objective,
        n_trials=FLAGS.n_trials,
        n_jobs=FLAGS.n_jobs,
        callbacks=[StopWhenTrialKeepBeingPrunedCallback(5)],
        catch=(ApiException,)
    )
    logger.info(f"Best trial: {study.best_trial.number}")
    logger.info(f"Best params: {study.best_trial.params}")

    best_job_id = f"{FLAGS.job_id}-{study.best_trial.number}"
    artifacts_dir = f"gs://{FLAGS.bucket}/artifacts/{FLAGS.training_pipeline_name}/{best_job_id}/training"
    with open("/tmp/out.txt", "w") as f:
        f.write(f"{artifacts_dir}/saved_model")


if __name__ == "__main__":
    app.run(main)
