# optuna-pipeline

AIモデル学習におけるハイパーパラメータの自動チューニングを行うためのパイプライン.

optunaとkubeflow pipelinesを組み合わせ、最適化を並列に実行できるようにしている.

実装はこちらを一部参照 => https://lab.mo-t.com/blog/optuna-kfp-hyperparameter-tuning

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
- [optuna-worker/main.go]()のcreate_settings関数の中に、目的の学習パイプライン用のハイパラチューニング設定を追加する

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
- チューニングしたいパラメータは、予め学習パイプラインの方でパイプラインパラメータ化しておく必要があります
- optuna-workerで各学習ジョブのHistoryからlossを見て最適化を行うように実装しているので、学習パイプライン側でもHistoryをartifactsとして保存しておく必要があります
- early stoppingを入れておくことを推奨します


## 実験結果の可視化

[optuna-dashboard](https://github.com/optuna/optuna-dashboard)を使用する.  
パラメータの寄与度や最適化の様子をグラフで確認できる.

### 手順

※将来的にkfpの一つのコンポーネントとして実装したいが、現状はできていない

、optunaで使用したDBをローカルPCにダンプしてきてインポートし、ローカルPCでoptuna-dashboardを起動する使い方になっている.


##### 1. 可視化したいoptuna-pipelineジョブが完了するのを待つ

##### 2. cloudSQLのインスタンスからダンプする
`データベース名：optuna`を指定

##### 3. ダンプしたファイルを手元のPCにダウンロードする

##### 4. mysqlにインポートする
```bash
$ mysql -u{ユーザ名} -p{パスワード} optuna < [dumpファイル名].sql
```

##### 5. optuna-dashboardを起動し出力されたURLにアクセスする

```bash
$ optuna-dashboard mysql+pymysql://{ユーザ名}:{パスワード}@localhost:3306/optuna
```

※study名はパイプライン実行時に指定したjob_idに一致する.
