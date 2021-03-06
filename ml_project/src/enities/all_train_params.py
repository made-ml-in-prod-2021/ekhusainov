from dataclasses import dataclass
from src.enities.train_test_split_parametrs import TrainTestSplitParametrs
from src.enities.feature_params import FeatureParams
from src.enities.model_params import ModelParams
from marshmallow_dataclass import class_schema
import yaml


@dataclass()
class TrainingPipelineParams:
    input_data_path: str
    output_model_path: str
    output_report_html: str
    path_to_one_hot_encoder: str
    path_to_scaler: str
    preprocessed_data_filepath: str
    x_test_filepath: str
    y_test_filepath: str
    y_train_filepath: str
    splitting_params: TrainTestSplitParametrs
    features_params: FeatureParams
    model_params: ModelParams


TrainingPipelineParamsSchema = class_schema(TrainingPipelineParams)


def read_training_pipeline_params(filepath: str) -> TrainingPipelineParams:
    with open(filepath, "r") as input_stream:
        schema = TrainingPipelineParamsSchema()
        return schema.load(yaml.safe_load(input_stream))
