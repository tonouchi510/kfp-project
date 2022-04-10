# kfp-project

Kubeflow Pipelinesで開発を始める際のテンプレートです。  
管理やCI/CDをやりやすくするためのディレクトリ設計になってます。

## パイプラインサンプル

GCPを前提にしており、本番サービス用のパイプラインはVertex Pipelines, 実験用パイプラインはAI-Platform Pipelinesを使う想定で実装例を用意しています.

#### vertex ai pipelines
- hello-world-pipeilne
  - 一番シンプルな実装例
- load-featurestore-pipeline
  - WIP
  - BigQueryからFeatureStoreにデータをロードするパイプライン
- online-training-pipeline
  - WIP
  - FeatureStoreからデータを取得し、学習するパイプライン

#### AIPlatform Pipelines
- head-pose-dataset-pipeline
  - vision apiを使って顔向き推定用のデータセットを作成するパイプライン
  - 他のラベルを公開apiで擬似アノテーションする際にもほぼ同様なので参考にしてください
- head-pose-pipeline
  - 上記顔向き推定データセットを使用して学習を行うパイプライン
  - tfでのトレーニングパイプラインの実装例として参考までに

vertexかaiplatformで若干コードやbuild.yamlが異なるので注意してください.

## パイプライン開発

パイプラインの一つのステップであるコンポーネントを開発し、それらの組み合わせでパイプラインを構築する。

この辺のリポジトリの実装例が参考になる.
- https://github.com/kubeflow/pipelines/tree/master/samples
- https://github.com/ksalama/kubeflow-examples/tree/master/kfp-cloudbuild

### ディレクトリ構成

```
$ tree -L 3 -d
.
├── components                      # 共通で使える汎用コンポーネントの置き場
│   ├── slack-notification          # slack通知を行うためのコンポーネント
│   └── tb-observer                 # TensorBoardを起動するためのコンポーネント
├── pipelines                       # 各種パイプラインの実装
│   ├── head-pose-dataset-pipeline
│   │   ├── data-chunk-spliter
│   │   └── pose-annotation
│   ├── head-pose-pipeline
│   │   ├── evaluation
│   │   └── training
│   ├── hello-world-pipeline
│   │   └── hello
│   ├── load-featurestore-pipeline
│   │   └── load
│   └── online-training-pipeline
│       └── training
└── utils                           # 共通の便利関数の置き場
```

#### pipelineのディレクトリ構成
パイプライン毎に、各種コンポーネントの実装とパイプラインの定義ファイルをおく。

```
Pipeline_X
├── Component_A          # コンポーネントのコード置き場
├── Component_B
├── settings.yaml        # 開発中にCIでデバッグ実行する際のパラメータを記述
└── xxx-pipeline.py      # パイプライン定義ファイル。kfpのDSLを使用して構築する。
```

#### componentの構成
コンポーネント毎にDockerイメージを用意して開発する。  
また、specファイルを活用してコンポーネントの仕様を定義する。

```
Pipeline_X
├── Component_A
│   ├── Dockerfile          # コンポーネントのイメージファイル
│   ├── component.yaml      # コンポーネントのspecファイル
│   ├── requirements.txt
│   └── main.py
└── Component_B
```

### CI/CD

GCPのAI Platform PipelinesもしくはVertex AI Pipelinesで実行することを前提としています。  
他の基盤にデプロイする場合は適宜yamlファイルを修正してください。

[こちらのリポジトリ](https://github.com/ksalama/kubeflow-examples/tree/master/kfp-cloudbuild)にある図のフローを参考にしている。
CloudBuild用に作られている部分をGitHub Actionsに変更している.  
パイプラインごとにyamlファイルを用意する必要があります.

![Pipeline CI/CD flow](https://github.com/ksalama/kubeflow-examples/raw/master/kfp-cloudbuild/resources/cloudbuild-steps.png)

## GCPサービス

以下のGCPサービスを使うことを想定している。サンプルコードは既にこれらのサービスにアクセスするので事前に有効化してください。

- AI Platform Pipelines: 実験用パイプラインの実行環境
- Vertex AI Pipelines: 本番サービス用パイプラインの実行環境
- CloudTPU: トレーニングパイプラインの効率化のために利用
- GCS: パイプラインの成果物・一時ファイルの保存に使用
- GCR: コンポーネントイメージのpush & pullに使用
- SecretManager: 実行時に必要となるSecretをここから取得する想定
- ServiceAccount: github actions実行用、パイプライン実行用に2つ必要
