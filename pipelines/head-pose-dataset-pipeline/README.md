# head-pose-dataset-pipeline

顔の向き推定モデルの学習データセットを作成するためのパイプライン.

GCPのvision APIを使ってhead-poseをアノテーションし、TFRecord形式で保存する.

### パイプラインパラメータ
- pipeline_name: str        # ジョブ管理用
- bucket_name: str          # ジョブ管理用
- job_id: str               # ジョブ管理用
- dataset: str              # 目的の画像ファイルが置いてあるフォルダ
- chunk_size: int           # 一つのTFRecordに含める画像ファイル数. `画像総数 / chunk_size`で並列実行される.
- valid_ratio: float        # validation用データを作る割合

### パイプライン成果物
- 擬似アノテーションされたデータセット（TFRecord形式）
