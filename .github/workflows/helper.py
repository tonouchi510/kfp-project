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
import kfp
import fire

HOST = "7450ddc5dd7a30f-dot-us-central1.pipelines.googleusercontent.com"
REPO_URL = "gcr.io/huroshotoku"
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
            print(f"Component {image_name} specs updated. Image: {full_image_name}")


def read_settings(pipeline_name: str, github_sha: str, is_master: bool):
    """Read all the parameter values from the settings.yaml file."""
    settings_file = os.path.join(PIPELINE_DIR, pipeline_name, SETTINGS_FILENAME)
    flat_settings = dict()
    setting_sections = yaml.safe_load(pathlib.Path(settings_file).read_text())
    for sections in setting_sections:
        if is_master:
            setting_sections[sections]["job_id"] = github_sha
        else:
            setting_sections[sections]["job_id"] = f"{pipeline_name}_{github_sha[:7]}"
            setting_sections[sections]["is_debug"] = True
        flat_settings.update(setting_sections[sections])
    return flat_settings


def deploy_pipeline(kfp_package_path: str, pipeline_name: str, github_sha: str, is_master: bool, run: bool):
    """Deploy and run the givne kfp_package_path."""

    client = kfp.Client(host=HOST)

    try:
        client.get_pipeline_id(name=pipeline_name if is_master else "debug")
        pipeline = client.upload_pipeline_version(
            pipeline_package_path=kfp_package_path,
            pipeline_version_name=github_sha if is_master else f"{pipeline_name}_version_at_{github_sha}",
            pipeline_name=pipeline_name if is_master else "debug")
    except Exception as e:
        print(e)
        print(f"creating {pipeline_name}.")
        pipeline = client.upload_pipeline(
            pipeline_package_path=kfp_package_path,
            pipeline_name=pipeline_name if is_master else "debug")
    pipeline_id = pipeline.id
    print(f"pipeline_id: {pipeline_id}")

    try:
        experiment = client.get_experiment(experiment_name=pipeline_name)
    except Exception as e:
        print(e)
        print(f"create experiment: {pipeline_name}.")
        experiment = client.create_experiment(name=pipeline_name)

    if run:
        run_id = f"Run of {pipeline_name}-{github_sha[:7]} by github actions"
        settings = read_settings(pipeline_name, github_sha, is_master)
        client.run_pipeline(
            experiment.id,
            job_name=run_id,
            version_id=pipeline_id,
            params=settings)


def main(operation, **args):
    # Update Component Specs
    if operation == 'update-specs':
        print('Setting images to the component spec...')
        image_tag = "latest"
        if 'image_tag' in args:
            image_tag = args['image_tag']
        update_component_spec(image_tag)

    # Deploy Pipeline
    elif operation == 'deploy-pipeline':
        print('Running Kubeflow pipeline...')
        if 'package_path' not in args:
            raise ValueError('package_path has to be supplied.')
        package_path = args['package_path']

        if 'pipeline_name' not in args:
            raise ValueError('pipeline_name has to be supplied.')
        pipeline_name = args['pipeline_name']

        if 'github_sha' not in args:
            raise ValueError('github_sha has to be supplied.')
        github_sha = args['github_sha']

        is_master = 'is_master' in args

        run = 'run' in args

        deploy_pipeline(package_path, pipeline_name, github_sha, is_master, run)

    # Check params
    elif operation == 'read-settings':
        print('Read pipeline params')
        if 'pipeline_name' not in args:
            raise ValueError('pipeline_name has to be supplied.')
        pipeline_dir = args['pipeline_name']

        if 'github_sha' not in args:
            raise ValueError('github_sha has to be supplied.')
        github_sha = args['github_sha']

        is_master = 'is_master' in args

        params = read_settings(pipeline_dir, github_sha, is_master)
        print(params)

    else:
        raise ValueError(
            'Invalid operation name: {}. Valid operations: update-specs | deploy-pipeline'.format(operation))


if __name__ == '__main__':
    fire.Fire(main)
