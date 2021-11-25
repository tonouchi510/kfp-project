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

import yaml
import os
import pathlib
import fire
import time
from kfp.v2.google.client import AIPlatformClient

PROJECT_ID = os.environ.get("GCP_PROJECT")
SERVICE_ACCOUNT_NAME = os.environ.get("VERTEX_SERVICE_ACCOUNT_NAME")
PIPELINE_DIR = "pipelines"


def update_component_spec(target_dir: str, image_tag: str):
    """Update the image in the component.yaml files.

    Args:
        target_dir (str): pipeline_dir.
            e.g. pipelines/xxxx-pipelie or components.
        image_tag (str): docker image tag.
    """
    for spec_path in pathlib.Path("components").glob("*/component.yaml"):
        spec = yaml.safe_load(pathlib.Path(spec_path).read_text())
        image_name = os.path.dirname(str(spec_path)).replace("components/", "")
        full_image_name = f"gcr.io/{PROJECT_ID}/{image_name}:latest"
        spec["implementation"]["container"]["image"] = full_image_name
        pathlib.Path(spec_path).write_text(yaml.dump(spec))
        print(f"Component {image_name} specs updated.")

    for spec_path in pathlib.Path(target_dir).glob("*/component.yaml"):
        spec = yaml.safe_load(pathlib.Path(spec_path).read_text())
        image_name = os.path.dirname(str(spec_path)).replace("pipelines/", "").replace("/", "-")
        full_image_name = f"gcr.io/{PROJECT_ID}/{image_name}:{image_tag}"
        spec["implementation"]["container"]["image"] = full_image_name
        pathlib.Path(spec_path).write_text(yaml.dump(spec))
        print(f"Component {image_name} specs updated. Image: {full_image_name}")


def read_settings(pipeline_name: str, version: str, is_debug: bool):
    """Read all the parameter values from the settings.yaml file."""
    if is_debug:
        job_id = f"debug-{version}"
        settings_file = os.path.join(PIPELINE_DIR, pipeline_name, "settings.debug.yaml")
    else:
        job_id = f"{pipeline_name}-{version}"
        settings_file = os.path.join(PIPELINE_DIR, pipeline_name, "settings.yaml")
    flat_settings = dict()
    setting_sections = yaml.safe_load(pathlib.Path(settings_file).read_text())
    for sections in setting_sections:
        setting_sections[sections]["job_id"] = job_id
        flat_settings.update(setting_sections[sections])
    return flat_settings


def run_pipeline(
    kfp_package_path: str,
    pipeline_name: str,
    version: str,
    is_debug: bool
) -> None:
    """Deploy and run the givne kfp_package_path."""
    t = int(time.time())
    job_id = f"debug-{version}-{t}" if is_debug else f"{pipeline_name}-{version}-{t}"
    settings = read_settings(pipeline_name, version, is_debug)

    print(f"Run {job_id}")
    try:
        client = AIPlatformClient(
            project_id=os.environ.get("GCP_PROJECT"), region=os.environ.get("GCP_REGION"))

        response = client.create_run_from_job_spec(
            kfp_package_path,
            job_id=job_id,
            pipeline_root=f"{os.environ.get('PIPELINE_ROOT')}/{version}",
            parameter_values=settings,
            service_account=SERVICE_ACCOUNT_NAME
        )
        print(response)
    except Exception as e:
        print(e)
        exit(-1)


def main(operation, **args):
    # Update Component Specs
    if operation == "update-specs":
        print("Setting images to the component spec...")
        image_tag = "latest"
        if "image_tag" in args:
            image_tag = args["image_tag"]

        if "target_dir" in args:
            target_dir = args["target_dir"]

        update_component_spec(target_dir, image_tag)

    # Run Pipeline
    elif operation == "run-pipeline":
        print("Running Kubeflow pipeline...")
        if "package_path" not in args:
            raise ValueError("package_path has to be supplied.")
        package_path = args["package_path"]

        if "pipeline_name" not in args:
            raise ValueError("pipeline_name has to be supplied.")
        pipeline_name = args["pipeline_name"]

        if "version" not in args:
            raise ValueError("version has to be supplied.")
        version = args["version"]

        is_debug = True if "debug" in args else False

        run_pipeline(package_path, pipeline_name, version, is_debug)

    # Check params
    elif operation == "read-settings":
        print("Read pipeline params")
        if "pipeline_name" not in args:
            raise ValueError("pipeline_name has to be supplied.")
        pipeline_dir = args["pipeline_name"]

        if "version" not in args:
            raise ValueError("github_sha has to be supplied.")
        version = args["version"]

        is_debug = True if "debug" in args else False

        params = read_settings(pipeline_dir, version, is_debug)
        print(params)

    else:
        raise ValueError(
            "Invalid operation name: {}. Valid operations: update-specs | run-pipeline".format(operation))


if __name__ == "__main__":
    fire.Fire(main)
