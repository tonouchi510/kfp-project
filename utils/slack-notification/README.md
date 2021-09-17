# slack notification

slackの指定したチャンネルにパイプラインの実行結果を送るためのコンポーネント。
事前にwebhook urlを発行し、設定する必要がある。

### 使用方法

パイプライン定義の際、`kfp.dsl.ExitHandler`を使用することでパイプラインジョブ終了時に必ず実行されるようにできる。

```python
slack_notification_op = component_store.load_component('slack-notification')

@dsl.pipeline(
    name='xxxx pipeline',
    description='hoge hoge fuga fuga'
)
def pipeline(
    pipeline_name: str = 'xxxx-pipeline',
    job_id: str = '{{JOB_ID}}'
):
    with dsl.ExitHandler(
        exit_op=slack_notification_op(
            pipeline=pipeline_name,
            bucket=bucket,
            jobid=job_id,
            message='Status: {{workflow.status}}'
        )
    ):
        xxxx_task = xxxx_op(
            ~~~~
        )
        ...
     

```