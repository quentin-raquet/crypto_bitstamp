import os
import logging
import yaml
import wandb
import torch
import torch.nn as nn
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from api_client import BitstampClient
from utils import str_to_datetime
from data import TimeSeriesDataModule
from model import TimeSeriesPredictionModel
from typing import Dict

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_NAME = "TCN-Crypto-Time-Series-Prediction"


def validate_config(config: Dict) -> Dict:
    """
    Validate the configuration dictionary.

    Args:
        config (Dict): The configuration dictionary.

    Returns:
        Dict: The validated configuration dictionary.
    """
    assert config["data"]["input_cols"][0] == "close", "close column is required in first position"
    config["data"]["context_len"] = sum([2**i for i in range(len(config["model"]["num_channels"]))])
    config["model"]["num_inputs"] = len(config["data"]["curr_symbs"]) * len(
        config["data"]["input_cols"]
    )
    config["model"]["num_outputs"] = len(config["data"]["curr_symbs"])
    config["data"]["start_date"] = str_to_datetime(config["data"]["start_date"])
    config["data"]["end_date"] = str_to_datetime(config["data"]["end_date"])
    return config


def prepare_data(client, data_config: Dict) -> TimeSeriesDataModule:
    """
    Prepare the data module.

    Args:
        data_config (Dict): The data configuration dictionary.

    Returns:
        TimeSeriesDataModule: The prepared data module.
    """
    data_module = TimeSeriesDataModule(
        client,
        data_config["curr_symbs"],
        data_config["context_len"],
        data_config["input_cols"],
        data_config["start_date"],
        data_config["end_date"],
        data_config["step_sec"],
        data_config["max_res"],
        data_config["batch_size"],
        data_config["val_size"],
        data_config["num_workers"],
    )
    data_module.setup()

    return data_module


def update_config(config: Dict, data_module: TimeSeriesDataModule) -> Dict:
    """
    Update the configuration dictionary with the data module parameters.

    Args:
        config (Dict): The configuration dictionary.
        data_module (TimeSeriesDataModule): The data module.

    Returns:
        Dict: The updated configuration dictionary.
    """
    config["model"]["min_X"] = data_module.min_X.tolist()
    config["model"]["max_X"] = data_module.max_X.tolist()
    config["model"]["min_y"] = data_module.min_y.tolist()
    config["model"]["max_y"] = data_module.max_y.tolist()
    return config


def train_model(config: Dict, data_module: TimeSeriesDataModule) -> None:
    """
    Train the model.

    Args:
        config (Dict): The configuration dictionary.
        data_module (TimeSeriesDataModule): The data module.
    """
    model = TimeSeriesPredictionModel(
        config["model"]["num_inputs"],
        config["model"]["num_channels"],
        config["model"]["num_outputs"],
        config["model"]["kernel_size"],
        config["model"]["dropout"],
        config["model"]["learning_rate"],
        config["model"]["min_y"],
        config["model"]["max_y"],
    )

    early_stopping_callback = EarlyStopping(
        monitor="val_mape", patience=config["model"]["patience"], mode="min"
    )
    model_checkpoint_callback = ModelCheckpoint(
        filename="model-{epoch:02d}-{val_mape:.6f}",
        monitor="val_mape",
        mode="min",
    )

    wdb_logger = WandbLogger(project=PROJECT_NAME, log_model="all")
    wdb_logger.log_hyperparams(config)

    trainer = Trainer(
        max_epochs=config["model"]["max_epochs"],
        callbacks=[early_stopping_callback, model_checkpoint_callback],
        logger=wdb_logger,
    )
    trainer.fit(model, datamodule=data_module)
    checkpoint_path = model_checkpoint_callback.best_model_path
    return checkpoint_path


def load_model(config: Dict, checkpoint_path: str) -> TimeSeriesPredictionModel:
    """
    Load the model from the checkpoint.

    Args:
        checkpoint_path (str): The path to the checkpoint file.

    Returns:
        TimeSeriesPredictionModel: The loaded
    """
    model = TimeSeriesPredictionModel(
        config["model"]["num_inputs"],
        config["model"]["num_channels"],
        config["model"]["num_outputs"],
        config["model"]["kernel_size"],
        config["model"]["dropout"],
        config["model"]["learning_rate"],
        config["model"]["min_y"],
        config["model"]["max_y"],
    )
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


class Pipeline(nn.Module):
    """
    A pipeline that preprocesses the input and passes it to the model.
    """

    def __init__(self, config: Dict, model: TimeSeriesPredictionModel):
        super(Pipeline, self).__init__()
        self.X_min = torch.tensor(config["model"]["min_X"], dtype=torch.float32)
        self.X_max = torch.tensor(config["model"]["max_X"], dtype=torch.float32)
        self.y_min = torch.tensor(config["model"]["min_y"], dtype=torch.float32)
        self.y_max = torch.tensor(config["model"]["max_y"], dtype=torch.float32)
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.X_min) / (self.X_max - self.X_min)
        y = self.model(x)
        y = y * (self.y_max - self.y_min) + self.y_min
        return y


def export_model_to_onnx(
    config: Dict, checkpoint_path: str
) -> None:
    """
    Export the model to ONNX format.

    Args:
        config (dict): The config dict.
        checkpoint_path (str): The path to the checkpoint file.
        output_path (str, optional): The path to the output directory. Defaults to "model_checkpoints".
    """
    model = load_model(config, checkpoint_path)

    pipeline = Pipeline(config, model)

    dummy_input = torch.randn(
        32,
        config["model"]["num_inputs"],
        sum([2**i for i in range(len(config["model"]["num_channels"]))]),
        requires_grad=True,
    )

    model_name = "model.onnx"

    torch.onnx.export(
        pipeline,
        dummy_input,
        model_name,
        export_params=True,
        opset_version=10,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )

    # Log the ONNX model to wandb
    artifact = wandb.Artifact(model_name, type="model")
    artifact.add_file(model_name)
    wandb.log_artifact(artifact)

    return


def main() -> None:
    """
    The main function that prepares the data, trains the model, and exports it to ONNX format.
    """
    with open("model/config.yaml", "r") as file:
        config = yaml.safe_load(file)
    config = validate_config(config)

    wandb.init(
        # set the wandb project where this run will be logged
        project="TCN-Crypto-Time-Series-Prediction",
        config=config,
    )

    logger.info("Start data preparation.")
    api_key = os.getenv("BITSTAMP_API_KEY")
    api_secret = os.getenv("BITSTAMP_API_SECRET")
    client = BitstampClient(api_key, api_secret)
    data_module = prepare_data(client, config["data"])
    config = update_config(config, data_module)
    logger.info("Data preparation done.")

    logger.info("Start training.")
    checkpoint_path = train_model(config, data_module)
    logger.info("Training completed.")

    logger.info("Export model to onnx.")
    export_model_to_onnx(config, checkpoint_path)
    logger.info("Model exported.")
    wandb.finish()


if __name__ == "__main__":
    main()
