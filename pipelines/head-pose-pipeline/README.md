# head-pose-pipeline

顔の向き（オイラー角）を推定するAIモデルの学習・評価パイプライン.

## Components

### training

head pose estimationをトレーニングするコンポーネント.  
現在はFSA-Netを実装済み. 教師なしのSSVも対応予定.

データセットは任意のTFRecord形式のファイルを読み込む.

### evaluation

test用データセットを使用して、 MAEやroll, pitch, yawごとの精度を算出する.
t
### 参照
- https://github.com/shamangary/FSA-Net
