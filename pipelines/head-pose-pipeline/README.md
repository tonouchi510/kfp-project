# head-pose-pipeline

顔の向き（オイラー角）を推定するAIモデルの学習・評価パイプライン.  
現状は[FSA-Net](https://github.com/shamangary/FSA-Net)を使用.

データセットはhead-pose-dataset-pipelineで作成すること.

### パイプラインパラメータ

- model_type: int		# fsa-netのモデルタイプ
- image_size: int		# 入力画像サイズ. 指定したサイズにリサイズする.
- dataset: str			# データセットのGCSパス. 空の場合は公開データセットを使用.
- test_dataset: str		# データセットのGCSパス. 空の場合は公開データセットを使用.

作成当時のSoTAモデルである[FSA-Net](https://github.com/shamangary/FSA-Net)の公開実装を引用し、AIPlatform Pipelines上で学習できる形にした.

### パイプライン成果物
- 顔向き推定の学習済みモデル(SavedModel形式)
- 精度評価の結果
  - MAEおよびyaw, pitch, rollごとのMAE
  - 推定結果の描画画像

## Components

### training

FSA-Netを学習するコンポーネント.  

データセットは任意のTFRecord形式のファイルを読み込むが、基本的にはhead-pose-datasetパイプラインで作成したデータセットを使用する.

パイプラインパラメータ: `dataset`に目的のTFRecordsが保存されているGCSパスを指定して使う.

### evaluation

評価用TFRecordを読み込んで顔向き推定結果のオイラー角のMAEやroll, pitch, yawごとの精度を算出する.  

head-pose-datasetパイプラインで作成されたデータセットはあくまでVision APIによる擬似アノテーションなので、人手でアノテーションされている公開データセットを使用して評価を行う. なお、[ここ](https://github.com/shamangary/FSA-Net#codes)の前処理済みデータセットがあるのでこれをTFRecord化して使用させてもらっている. 必要なデータをGCSにアップロードして使用してください.

また、確認用に推定結果を描画した画像も一部生成し、artifactsフォルダに保存しているので、予測結果の目視チェックを行いたい場合はそちらを確認してください.
=>
```gs://kfp-project/artifacts/head-pose-pipeline/{job_id}/evaluation/imgs/```
