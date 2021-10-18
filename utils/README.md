# utilsパッケージ

共通の関数や定義など。その他実験用の各種helper関数も。

## trainer

Deepモデル作成における共通ライブラリ。
kfpでのトレーニングパイプラインを効率的に開発できるようにする。

- プリエンプト対応のためのチェックポイント作成 & 復旧
- CloudTPUでのトレーニング
- tf.keras.Modelsに寄せた共通化
  - こちらに記載の内容と同様
  - https://github.com/tonouchi510/tensorflow-design

ただし、フレームワーク：TensorFlow2、データ形式：TFRecord、その他以下のフォルダ設計を前提とする。

```
gs://{{bucket_name}}
├── datasets							# データセットを管理するフォルダ
│   ├── ○○_datasets						  - tfrecordを細かいチャンクに区切って保存する
│   │   ├── train_00001.tfrec
│   │   ├── train_00002.tfrec
│   │   ├── valid_00001.tfrec
│   │   ├── valid_00002.tfrec
│   │   └── label.txt					  # クラスを記述したファイルが必要
│   └── xxx_datasets
├── tmp									# パイプラインジョブの一時ファイル置き場
│   └── {{pipeline_name}}				  - `パイプライン名/ジョブID/コンポーネント名`
│       ├── {{job_id}}						ごとに管理する。
│       │   ├── {{component_name}}
│       │   └── {{component_name}}
│       └── {{job_id}}
└── artifacts							# パイプラインの成果物置き場
    └── {{pipeline_name}}
    │   ├── {{job_id}}
    │   │   └── {{component_name}}
    │   └── {{job_id}}
    └── {{pipeline_name}}

```

