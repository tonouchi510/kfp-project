# hello-world-pipeline

とりあえずチュートリアル的な簡単なサンプル。

メッセージが記述されたファイルを読み込み、メッセージの数分並列して標準出力するパイプラインです。


#### パイプラインパラメータ

```
bucket: メッセージファイルが保存されたバケット
job_id: パイプラインジョブのID
message_file: メッセージが記述されたファイルのGCSパス
```

### パイプラインの概要

図