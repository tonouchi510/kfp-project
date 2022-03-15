# head-pose-pipeline

顔の向き（オイラー角）を推定するAIモデルの学習・評価パイプライン.  
現状は[FSA-Net](https://github.com/shamangary/FSA-Net)を使用.

データセットはhead-pose-dataset-pipelineで作成すること.

## パイプラインパラメータ

- model_type: int		# fsa-netのモデルタイプ
- image_size: int		# 入力画像サイズ. 指定したサイズにリサイズする.
- dataset: str			# データセットのGCSパス. 空の場合は公開データセットを使用.
- test_dataset: str		# データセットのGCSパス. 空の場合は公開データセットを使用.

作成当時のSoTAモデルである[FSA-Net](https://github.com/shamangary/FSA-Net)の公開実装を引用し、AIPlatform Pipelines上で学習できる形にした.

## Components

### training

FSA-Netを学習するコンポーネント.  

データセットは任意のTFRecord形式のファイルを読み込むが、基本的にはhead-pose-datasetパイプラインで作成したデータセットを使用する.

パイプラインパラメータ: `dataset`に目的のTFRecordsが保存されているGCSパスを指定して使う.

### evaluation

head-pose-datasetパイプラインで作成されたデータセットはあくまでVision APIによる擬似アノテーションなので、人手でアノテーションされている公開データセットを使用して、MAEやroll, pitch, yawごとの精度を算出する.

[ここ](https://github.com/shamangary/FSA-Net#codes)の前処理済みデータセットがあるのでこれを使用させてもらっている. 必要なデータをGCSにダウンロードして使用してください.

例えば、データセットはダウンロードして`gs://{YOUR_BUCKET_NAME}/datasets/{dataset_name}/`以下に置くこと.
