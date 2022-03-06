import json
import logging
from typing import List

logger = logging.getLogger(__name__)

# components.yamlにも下記と同名でアウトプットを記述すること
METRICS_FILE = "/mlpipeline-metrics.json"
UI_METADATA_FILE = "/mlpipeline-ui-metadata.json"


class KFPVisualization(object):
    """Kubeflow Pipelinesの可視化機能を使用するためのクラス.

    UI:
        https://www.kubeflow.org/docs/components/pipelines/sdk/output-viewer/
    Metrics:
        https://www.kubeflow.org/docs/components/pipelines/sdk/pipelines-metrics/
    """
    def __init__(
        self,
        pipeline_name: str,
        bucket_name: str,
        job_id: str
    ):
        self.job_dir: str = f"gs://{bucket_name}/tmp/{pipeline_name}/{job_id}"
        self.metrics: List = []
        self.metadata_outputs: List = []

    def produce_metrics(
        self,
        name: str,
        value: float,
        f: str = "PERCENTAGE"
    ):
        """kubeflowのui metricsのフォーマットに整形し、登録する.

        ※複数ある場合は一つずつ呼び出し、実行する.

        Args:
            name (str): メトリクス名.
            value (float): 表示する値.
            f (str, optional): format. Defaults to "PERCENTAGE".
        """
        metrics = {
            'name': name,
            'numberValue': value,
            'format': f,
        }
        self.metrics.append(metrics)

    def write_metrics(self):
        """登録済みのmetricsを可視化用にファイルに書き込む.

        注: 複数のmetricsがある場合は全て登録後に実行すること.
        """
        metrics = {
            'metrics': self.metrics
        }
        with open(METRICS_FILE, 'w') as f:
            json.dump(metrics, f)

    def produce_ui_metadata_table(
        self,
        source: str,
        header: List,
        storage: str = "gcs",
        f: str = "csv"
    ):
        """kubeflowのpipeline-ui-metadataのフォーマットに整形し、登録する.
        
        ※複数ある場合は一つずつ呼び出し、実行する.

        Args:
            source (List): tableに表示するデータ.
            storage (str, optional): [description]. Defaults to "gcs".
            f (str, optional): [description]. Defaults to "csv".
        """
        output = {
            'type': 'table',
            'storage': storage,
            'format': f,
            'header': header,
            'source': source
        }
        self.metadata_outputs.append(output)

    def write_ui_metadata(self):
        """登録済みのui metadataを可視化用にファイルに書き込む.

        注: 複数のui metadataがある場合は全て登録後に実行すること.
        """
        metadata = {
            'outputs': self.metadata_outputs
        }
        with open(UI_METADATA_FILE, 'w') as f:
            json.dump(metadata, f)
