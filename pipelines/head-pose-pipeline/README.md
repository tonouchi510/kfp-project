# head-pose-pipeline

顔の向き（オイラー角）を推定するAIモデルの学習・評価パイプライン.

作成当時のSoTAモデルである[FSA-Net](https://github.com/shamangary/FSA-Net)の公開実装を引用し、AIPlatform Pipelines上で学習できる形にした.

## Components

### training

FSA-Netを学習するコンポーネント.  

データセットは任意のTFRecord形式のファイルを読み込むが、基本的にはhead-pose-datasetパイプラインで作成したデータセットを使用する.

パイプラインパラメータ: `dataset`に目的のTFRecordsが保存されているGCSパスを指定して使う.

### evaluation

head-pose-datasetパイプラインで作成されたデータセットはあくまでVision APIによる擬似アノテーションなので、人手でアノテーションされている公開データセットを使用して、MAEやroll, pitch, yawごとの精度を算出する.

現状はBIWIデータセットのみ対応している. [ここ](https://github.com/shamangary/FSA-Net#codes)を見て必要なデータをダウンロードして使用してください.

例えばBIWIデータセットはダウンロードして`gs://{YOUR_BUCKET_NAME}/datasets/BIWI/`以下に置くこと.
