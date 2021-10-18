# kfp-project

Kubeflow Pipelinesで開発を始める際のテンプレートです。  
管理やCI/CDをやりやすくするためのディレクトリ設計になってます。

## パイプライン開発

パイプラインの一つのステップであるコンポーネントを開発し、それらの組み合わせでパイプラインを構築する。

### ディレクトリ構成

```
$ tree -L 3 -d
.
├── base                            # baseにするdockerイメージ置き場。共通化や本番環境との統一化が目的。
├── configs                         # 設定ファイル置き場
├── components                      # 共通で使える汎用コンポーネントの置き場
│   ├── slack-notification            # slack通知を行うためのコンポーネント
│   └── tb-observer                   # TensorBoardを起動するためのコンポーネント
├── pipelines                       # 各種パイプラインの実装
│   ├── hello-world-pipeline          # パイプラインの定義やパイプライン固有のコンポーネントの実装
│   │   └── hello
│   ├── dataset-pipeline
│   │   ├── tfrecord-converter
│   │   └── parallelizer
│   └── training-pipeline
│       └── training
├── tests
└── utils                           # 共通の便利関数の置き場
```

#### pipelineのディレクトリ構成
パイプライン毎に、各種コンポーネントの実装とパイプラインの定義ファイルをおく。

```
Pipeline_X
├── Component_A          # コンポーネントのコード置き場
├── Component_B
├── settings.debug.yaml  # 開発中にCIで実行する際のデバッグ実行用パラメータを記述
├── settings.yaml        # masterブランチマージ時にCIで実行する際のパラメータを記述
└── xxx-pipeline.py      # パイプライン定義ファイル。kfpのDSLを使用して構築する。
```

#### componentの構成
コンポーネント毎にDockerイメージを用意して開発する。`base`ディレクトリにあるイメージを元にして作成する。  
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

GCPのVertex AI Pipelinesで実行することを前提としています。多少修正すれば他のインフラにデプロイすることも可能です。

[こちらのリポジトリ](https://github.com/ksalama/kubeflow-examples/tree/master/kfp-cloudbuild)にある図のフローを参考にしている。
CloudBuild用に作られている部分をGitHub Actionsに変更している。また、複数パイプラインのデプロイに対応させています。

![Pipeline CI/CD flow](https://github.com/ksalama/kubeflow-examples/raw/master/kfp-cloudbuild/resources/cloudbuild-steps.png)

※注意：
- 複数パイプラインをこのリポジトリで一元管理している都合上、どのパイプライン・コンポーネントをRebuildするか制御するために、現状では`.github/workflows/deploy-targets.txt`で指定する必要がある
- 将来的にもう少し便利なビルド制御を行えるようにしたい

## GCPサービス

以下のGCPサービスを使うことを想定している。サンプルコードは既にこれらのサービスにアクセスするので事前に有効化してください。

- Vertex AI Pipelines: ここで構築したパイプラインの実行環境
- CloudTPU: トレーニングパイプラインの効率化のために利用
- GCS: パイプラインの成果物・一時ファイルの保存に使用
- GCR: コンポーネントイメージのpush & pullに使用
- SecretManager: 実行時に必要となるSecretをここから取得する想定
  - 現状はslack-notificationコンポーネントでwebhook url取得するために使ってる
- ServiceAccount: github actions実行用、パイプライン実行用に2つ必要
