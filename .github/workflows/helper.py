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
from kfp.v2.google.client import AIPlatformClient
from utils.logger import get_logger

logger = get_logger(__name__)

REPO_URL = f"gcr.io/{os.environ.get('GCP_PROJECT_ID')}"
PIPELINE_DIR = "pipelines"
SETTINGS_FILENAME = "settings.yaml"


def update_component_spec(image_tag: str):
    """Update the image in the component.yaml files given the repo_url and image_tag."""

    targets = pathlib.Path("pipelines").glob("*-pipeline")
    for target_dir in targets:
        for spec_path in pathlib.Path(target_dir).glob("*/component.yaml"):
            spec = yaml.safe_load(pathlib.Path(spec_path).read_text())
            image_name = os.path.dirname(str(spec_path)).replace("pipelines/", "").replace("/", "-")
            full_image_name = f"{REPO_URL}/{image_name}:{image_tag}"
            spec["implementation"]["container"]["image"] = full_image_name
            pathlib.Path(spec_path).write_text(yaml.dump(spec))
            logger.info(f"Component {image_name} specs updated. Image: {full_image_name}")


def read_settings(pipeline_name: str, github_sha: str):
    """Read all the parameter values from the settings.yaml file."""
    settings_file = os.path.join(PIPELINE_DIR, pipeline_name, SETTINGS_FILENAME)
    flat_settings = dict()
    setting_sections = yaml.safe_load(pathlib.Path(settings_file).read_text())
    for sections in setting_sections:
        setting_sections[sections]["job_id"] = github_sha
        #setting_sections[sections]["job_id"] = f"{pipeline_name}_{github_sha[:7]}"
        #setting_sections[sections]["is_debug"] = True
        flat_settings.update(setting_sections[sections])
    return flat_settings


def run_pipeline(
    kfp_package_path: str,
    pipeline_name: str,
    github_sha: str,
    datetime: str
) -> None:
    """Deploy and run the givne kfp_package_path."""

    client = AIPlatformClient(
        project_id=os.environ.get("GCP_PROJECT_ID"), region=os.environ.get("GCP_REGION"))
    settings = read_settings(pipeline_name, github_sha)

    try:
        response = client.create_run_from_job_spec(
            kfp_package_path,
            pipeline_root=f"{os.environ.get('PIPELINE_ROOT')}/{datetime}",
            parameter_values=settings
        )
        logger.info(response)
    except Exception as e:
        logger.error(e)


def main(operation, **args):
    # Update Component Specs
    if operation == "update-specs":
        logger.info("Setting images to the component spec...")
        image_tag = "latest"
        if "image_tag" in args:
            image_tag = args["image_tag"]
        update_component_spec(image_tag)

    # Run Pipeline
    elif operation == "run-pipeline":
        logger.info('Running Kubeflow pipeline...')
        if "package_path" not in args:
            raise ValueError("package_path has to be supplied.")
        package_path = args["package_path"]

        if "pipeline_name" not in args:
            raise ValueError("pipeline_name has to be supplied.")
        pipeline_name = args["pipeline_name"]

        if "github_sha" not in args:
            raise ValueError("github_sha has to be supplied.")
        github_sha = args["github_sha"]

        if "datetime" not in args:
            raise ValueError("datetime has to be supplied.")
        github_sha = args["datetime"]

        run_pipeline(package_path, pipeline_name, github_sha)

    # Check params
    elif operation == 'read-settings':
        logger.info('Read pipeline params')
        if 'pipeline_name' not in args:
            raise ValueError('pipeline_name has to be supplied.')
        pipeline_dir = args['pipeline_name']

        if 'github_sha' not in args:
            raise ValueError('github_sha has to be supplied.')
        github_sha = args['github_sha']

        params = read_settings(pipeline_dir, github_sha)
        logger.debug(params)

    else:
        raise ValueError(
            'Invalid operation name: {}. Valid operations: update-specs | deploy-pipeline'.format(operation))


if __name__ == '__main__':
    fire.Fire(main)
