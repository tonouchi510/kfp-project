import ast
from google.cloud import secretmanager


def access_secret(
    project_id: str,
    secret_id: str,
    version_id: str = "latest"
):
    """
    Access the payload for the given secret version if one exists. The version
    can be a version number as a string (e.g. "5") or an alias (e.g. "latest").
    """

    client = secretmanager.SecretManagerServiceClient()

    name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"
    response = client.access_secret_version(request={"name": name})

    payload = response.payload.data.decode("UTF-8")
    res = ast.literal_eval(payload)
    return res

