# tensorboard observer

tensorboard起動用のコンポーネント。

kubeflowの仕様上、tensorboardポッドの起動はコンポーネントが終了するまでできない。
これではトレーニング途中で実験経過を確認したい場合に不都合となる。

そのため、トレーニングコンポーネント内ではtensorboardのログ出力だけに止めておき、
別でtensorboardポッド起動用のコンポーネントを立てておき、そこからtensorboardログの出力先を監視するようにした。
これにより、トレーニング途中でも、こちらのコンポーネントからtensorboardを起動できる。
