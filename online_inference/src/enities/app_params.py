from dataclasses import dataclass
from marshmallow_dataclass import class_schema

import yaml

DEFAULT_CONFIG_APP = "configs/app_params.yml"


@dataclass()
class AppParams:
    ip_inside_docker: str
    url_external: str
    port: int
    data_for_predict_path: str


AppParamsSchema = class_schema(AppParams)


def read_app_params(filepath: str = DEFAULT_CONFIG_APP) -> AppParams:
    with open(filepath, "r") as input_schema:
        schema = AppParamsSchema()
        return schema.load(yaml.safe_load(input_schema))
