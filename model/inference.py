import re
import yaml
import torch
import torch.nn as nn
from model import TimeSeriesPredictionModel


def load_model(hparams_path, checkpoint_path):

    with open(hparams_path, "r") as file:
        hparams = yaml.safe_load(file)
    model = TimeSeriesPredictionModel(**hparams)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, hparams


def extract_model_info(checkpoint_path):
    match = re.search(r'model-epoch=(\d+)-val_loss=(\d+\.\d+)', checkpoint_path)
    if match:
        model_epoch = match.group(1)
        val_loss = match.group(2)
        return model_epoch, val_loss
    else:
        raise ValueError("Invalid checkpoint path. Could not extract model-epoch and val_loss.")


class Pipeline(nn.Module):
    def __init__(self, hparams, model):
        super(Pipeline, self).__init__()
        self.X_min = hparams["X_min"]
        self.X_max = hparams["X_max"]
        self.y_min = hparams["y_min"]
        self.y_max = hparams["y_max"]
        self.model = model

    def forward(self, x):
        x = (x - self.X_min) / (self.X_max - self.X_min)
        y = self.model(x)
        y = y * (self.y_max - self.y_min) + self.y_min
        return y


def export_model_to_onnx(hparams_path, checkpoint_path, output_path="model_checkpoints"):
    model, hparams = load_model(hparams_path, checkpoint_path)
    model_epoch, val_loss = extract_model_info(checkpoint_path)

    # Add the preprocessing step to the model
    pipeline = Pipeline(hparams, model)

    dummy_input = torch.randn(
        32,  # batch_size
        hparams["num_inputs"],
        sum([2**i for i in range(len(hparams["num_channels"]))]),
        requires_grad=True,
    )

    torch.onnx.export(
        pipeline,
        dummy_input,
        output_path + f"/model-epoch={model_epoch}-val_loss={val_loss}.onnx",
        export_params=True,
        opset_version=10,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    return


export_model_to_onnx(
    "/home/queraq/crypto_bitstamp/logs/model_logs/version_1/hparams.yaml",
    "/home/queraq/crypto_bitstamp/model_checkpoints/model-epoch=09-val_loss=0.007548.ckpt"
)