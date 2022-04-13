# optuna-pipeline

AIモデル学習におけるハイパーパラメータの自動チューニングを行うためのパイプライン.

optunaとkubeflow pipelinesを組み合わせ、最適化を並列に実行できるようにしている.

## 使い方

何らかのAIモデル学習用のパイプラインが既にあり、ハイパラチューニングが必要な場合は利用推奨。

### パイプラインパラメータ

- pipeline_name: str            # ジョブ管理用. optuna-pipeline or debug.
- bucket_name: str              # ジョブ管理用. 
- job_id: str                   # ジョブ管理用の一意のID.
- n_trials: int                 # 最適化のための学習ジョブの試行回数.
- n_jobs: int                   # 学習ジョブの並列実行数.
- training_pipeline_name: str   # ハイパラチューニング対象(optuna-pipelineから起動する)の学習パイプラインの名前.
- dataset: str                  # 使用するデータセットのGCSパス.
- epochs: int                   # 各学習ジョブのepoch数.

### 事前コーディング

- 目的の学習パイプラインを用意する
- `optuna-worker/main.go#create_settings`関数の中に、目的の学習パイプライン用のハイパラチューニング設定を追加する

```python:pipelines/head-pose-pipelineの例
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
```

### 実行

- ハイパーパラメータの`training_pipeline_name`を目的のパイプライン名にしていすることで、そのパイプラインのデプロイされている最新バージョンが実行される
- 他の各種パラメータも目的の値に変更し、実行すればOK

### トレーニングパイプラインの実装注意
- チューニングしたいパラメータは、予め学習パイプラインの方でパイプラインパラメータ化しておく必要がある
- optuna-workerで各学習ジョブのHistoryからlossを見て最適化を行うように実装しているので、学習パイプライン側でもHistoryをartifactsとして保存しておく必要がある
- early stoppingを入れておくことを推奨
