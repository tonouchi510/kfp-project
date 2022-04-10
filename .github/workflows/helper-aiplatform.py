# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Helper module to deploy and run pipelines."""

import os
import yaml
from pathlib import Path
import fire
import kfp

PROJECT_ID = os.environ.get("GCP_PROJECT")
KUBEFLOW_HOST = os.environ.get("KUBEFLOW_HOST")
SERVICE_ACCOUNT_NAME = os.environ.get("AIP_SERVICE_ACCOUNT_NAME")
PIPELINE_DIR = "pipelines"


def update_component_spec(target_dir: str, image_tag: str):
    """Update the image in the component.yaml files.
    Args:
        target_dir (str): pipeline_dir, e.g. pipelines/xxxx-pipelie.
        image_tag (str): docker image tag.
    """

    for spec_path in Path("components").glob("*/component.yaml"):
        spec = yaml.safe_load(Path(spec_path).read_text())
        image_name = os.path.dirname(str(spec_path)).replace("components/", "")
        full_image_name = f"gcr.io/{PROJECT_ID}/{image_name}:latest"
        spec["implementation"]["container"]["image"] = full_image_name
        Path(spec_path).write_text(yaml.dump(spec))
        print(f"Component {image_name} specs updated.")

    for spec_path in Path(target_dir).glob("*/component.yaml"):
        spec = yaml.safe_load(Path(spec_path).read_text())
        image_name = os.path.dirname(str(spec_path)).replace("pipelines/", "").replace("/", "-")
        full_image_name = f"gcr.io/{PROJECT_ID}/{image_name}:{image_tag}"
        spec["implementation"]["container"]["image"] = full_image_name
        Path(spec_path).write_text(yaml.dump(spec))
        print(f"Component {image_name} specs updated. Image: {full_image_name}")


def read_settings(pipeline_name: str, version: str, is_master: bool):
    """Read all the parameter values from the settings.yaml file.
    Args:
        pipeline_name (str): パイプライン名.
        version (str): パイプラインのバージョン.
        is_master (bool): masterブランチならTrue.
    Returns:
        [type]: [description]
    """
    settings_file = os.path.join(PIPELINE_DIR, pipeline_name, "settings.yaml")
    flat_settings = dict()
    setting_sections = yaml.safe_load(Path(settings_file).read_text())
    for sections in setting_sections:
        setting_sections[sections]["job_id"] = version
        if not is_master:
            setting_sections[sections]["job_id"] = f"{pipeline_name}_{version}"
            setting_sections[sections]["pipeline_name"] = "debug"
        flat_settings.update(setting_sections[sections])
    return flat_settings


def deploy_pipeline(pipeline_name: str, version: str, is_master: bool, run: bool):
    """Deploy and run the givne kfp_package_path."""

    client = kfp.Client(host=KUBEFLOW_HOST)

    try:
        client.get_pipeline_id(name=pipeline_name if is_master else "debug")
        pipeline = client.upload_pipeline_version(
            pipeline_package_path=f"{pipeline_name}.tar.gz",
            pipeline_version_name=version if is_master else f"{pipeline_name}_version_at_{version}",
            pipeline_name=pipeline_name if is_master else "debug")
    except Exception as e:
        print(e)
        print(f"creating {pipeline_name}.")
        pipeline = client.upload_pipeline(
            pipeline_package_path=f"{pipeline_name}.tar.gz",
            pipeline_name=pipeline_name if is_master else "debug")
    pipeline_id = pipeline.id
    print(f"pipeline_id: {pipeline_id}")

    try:
        experiment = client.get_experiment(experiment_name=pipeline_name if is_master else "debug")
    except Exception as e:
        print(e)
        print(f"creating experiment: {pipeline_name if is_master else 'debug'}.")
        experiment = client.create_experiment(name=pipeline_name if is_master else "debug")

    if run:
        run_id = f"Run of {pipeline_name}-{version}"
        settings = read_settings(pipeline_name, version, is_master)
        client.run_pipeline(
            experiment.id,
            job_name=run_id,
            version_id=pipeline_id,
            params=settings)


def main(operation, **args):

    # Update Component Specs
    if operation == "update-specs":
        print("Setting images to the component spec...")

        if "target_dir" not in args:
            raise ValueError("target_dir has to be supplied.")
        target_dir = args["target_dir"]

        image_tag = "latest"
        if "image_tag" in args:
            image_tag = args["image_tag"]
        update_component_spec(target_dir, image_tag)

    # Deploy Pipeline
    elif operation == "deploy-pipeline":
        print("Running Kubeflow pipeline...")

        if "pipeline_name" not in args:
            raise ValueError("pipeline_name has to be supplied.")
        pipeline_name = args["pipeline_name"]

        if "version" not in args:
            raise ValueError("version has to be supplied.")
        version = args["version"]

        is_master = "is_master" in args

        run = "run" in args

        deploy_pipeline(pipeline_name, version, is_master, run)

    # Check params
    elif operation == "read-settings":
        print("Read pipeline params")
        if "pipeline_name" not in args:
            raise ValueError("pipeline_name has to be supplied.")
        pipeline_dir = args["pipeline_name"]

        if "version" not in args:
            raise ValueError("version has to be supplied.")
        version = args["version"]

        is_master = "is_master" in args

        params = read_settings(pipeline_dir, version, is_master)
        print(params)

    else:
        raise ValueError(
            "Invalid operation name: {}. Valid operations: update-specs | deploy-pipeline".format(operation))


if __name__ == "__main__":
    fire.Fire(main)
