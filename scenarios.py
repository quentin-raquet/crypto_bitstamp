import os
import yaml
import torch
from datetime import datetime
from tqdm import tqdm
import numpy as np
import pandas as pd
import hydra
import logging
from omegaconf import DictConfig
from chronos import ChronosPipeline
from utils import get_history, generate_time_intervals, str_to_datetime
from api_client import BitstampClient
from typing import Any, Dict, List
from torch import Tensor

from torchmetrics.functional import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score,
)

logger = logging.getLogger(__name__)


def get_ohlc_df(
    client: BitstampClient,
    start_date: datetime,
    end_date: datetime,
    step_sec: int,
    max_res: int,
    curr_symb: str,
) -> pd.DataFrame:
    """
    Retrieves OHLC (Open-High-Low-Close) data from the client for the specified time range.

    Args:
        client: The Bitstamp client.
        start_date: The start date of the data range.
        end_date: The end date of the data range.
        step_sec: The time interval in seconds.
        max_res: The maximum number of data points to retrieve.
        curr_symb: The currency symbol.

    Returns:
        The OHLC data as a pandas DataFrame.
    """
    logger.info(f"Retrieve OHLC data from client: {curr_symb} from {start_date} to {end_date}")
    intervals = generate_time_intervals(start_date, end_date, step_sec, max_res)

    columns = ["timestamp", "open", "high", "low", "close", "volume", "datetime"]
    ohlc_df = pd.DataFrame({}, columns=columns).reindex(columns=columns)

    for start_dt, end_dt in tqdm(intervals):
        ohlc_df = pd.concat(
            [
                ohlc_df,
                get_history(
                    client,
                    curr_symb,
                    step_sec=step_sec,
                    limit=max_res,
                    start_dt=start_dt,
                    end_dt=end_dt,
                ),
            ],
            ignore_index=True,
        )

    return ohlc_df


def build_context(ohlc_df: pd.DataFrame, context_len: int, cols: List[str]) -> Tensor:
    """
    Builds the context tensor for the model.

    Args:
        ohlc_df: The OHLC data as a pandas DataFrame.
        context_len: The length of the context.
        cols: The columns to include in the context.

    Returns:
        The context tensor.
    """
    logger.info("Build context tensor for model")
    output_len = len(ohlc_df) - context_len
    context = np.empty((output_len, len(cols), context_len))
    for i in tqdm(range(context_len, len(ohlc_df) - 1)):
        context[i - context_len] = ohlc_df.loc[(i - context_len) : (i - 1), cols].values.T
    return torch.tensor(context)


def build_target(ohlc_df: pd.DataFrame, context_len: int) -> Tensor:
    """
    Builds the target tensor for the model.

    Args:
        ohlc_df: The OHLC data as a pandas DataFrame.
        context_len: The length of the context.

    Returns:
        The target tensor.
    """
    return torch.tensor(ohlc_df.loc[context_len:, "close"].values)


def predict_prices(context: Tensor, model_name: str, num_samples: int = 5) -> Tensor:
    """
    Predicts the prices using the trained model.

    Args:
        context: The context tensor.
        model_name: The name of the trained model.
        num_samples: The number of samples to generate for each prediction.

    Returns:
        The predicted prices tensor.
    """
    logger.info("Predict prices using trained model")
    pipeline = ChronosPipeline.from_pretrained(
        model_name,
        device_map="cpu",  # use "cpu" for CPU inference or "cuda"
        torch_dtype=torch.bfloat16,
    )
    predictions = torch.empty(context.shape[0], 2)
    for i in tqdm(range(context.shape[0])):
        forecast = pipeline.predict(
            context=context[i],
            prediction_length=1,
            num_samples=num_samples,
        ).squeeze()
        if forecast.dim() == 1:
            forecast = forecast.unsqueeze(0)
        mean = forecast[0, :].mean().item()
        std = forecast[0, :].std().item()
        if abs(mean) > 1e100:
            mean = 0.0
            logger.warning("Mean is too large")
        if abs(std) < 1e-6:
            logger.warning("Std is too small")
        predictions[i, 0] = mean
        predictions[i, 1] = std

    return predictions


def compute_metrics(preds: Tensor, targets: Tensor) -> Dict[str, float]:
    """
    Computes evaluation metrics for the predicted prices.

    Args:
        preds: The predicted prices tensor.
        targets: The target prices tensor.

    Returns:
        A dictionary containing the computed metrics.
    """
    metrics = {
        "MAE": mean_absolute_error(preds, targets).item(),
        "MSE": mean_squared_error(preds, targets).item(),
        "RMSE": torch.sqrt(mean_squared_error(preds, targets)).item(),
        "MAPE": mean_absolute_percentage_error(preds, targets).item(),
        "R2": r2_score(preds, targets).item(),
    }
    logger.info("Metrics computed")
    return metrics


def build_output(
    ohlc_df: pd.DataFrame, context_len: int, predictions: Tensor, targets: Tensor
) -> pd.DataFrame:
    """
    Builds the output DataFrame containing the target and predicted prices.

    Args:
        ohlc_df: The OHLC data as a pandas DataFrame.
        context_len: The length of the context.
        predictions: The predicted prices tensor.
        targets: The target prices tensor.

    Returns:
        The output DataFrame.
    """
    output = pd.DataFrame(
        {"target": targets, "prediction": predictions[:, 0]},
        index=ohlc_df.loc[context_len:, "datetime"],
    )
    return output


def export_results(output_folder: str, output_df: pd.DataFrame, metrics: Dict[str, float]) -> None:
    """
    Exports the results to the specified output folder.

    Args:
        output_folder: The output folder path.
        output_df: The output DataFrame.
        metrics: The computed metrics.
    """
    logger.info("Export results to output folder")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_df.to_csv(output_folder + "/output.csv")
    with open(output_folder + "/metrics.yml", "w") as f:
        yaml.dump(metrics, f)


def validate_config(config: Dict[str, Any]) -> None:
    assert config.api.start_date < config.api.end_date, "Start date must be before end date"
    assert config.model.context_len > 0, "Context length must be greater than 0"
    assert config.api.step_sec > 0, "Step seconds must be greater than 0"
    limit = (
        str_to_datetime(config.api.end_date) - str_to_datetime(config.api.start_date)
    ).total_seconds() / config.api.step_sec
    assert config.model.context_len < limit, "Context length must be less than max resolution"
    logger.info("Config validated")


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(config: DictConfig) -> None:
    """
    The main function that orchestrates the data retrieval, model prediction, and result export.

    Args:
        config: The configuration dictionary.
    """
    validate_config(config)
    api_key = os.getenv("BITSTAMP_API_KEY")
    api_secret = os.getenv("BITSTAMP_API_SECRET")
    client = BitstampClient(api_key, api_secret)
    start_date = str_to_datetime(config.api.start_date)
    end_date = str_to_datetime(config.api.end_date)
    ohlc_df = get_ohlc_df(
        client,
        start_date,
        end_date,
        config.api.step_sec,
        config.api.max_res,
        config.api.curr_symb,
    )
    context = build_context(ohlc_df, config.model.context_len, config.model.cols)
    targets = build_target(ohlc_df, config.model.context_len)
    predictions = predict_prices(context, config.model.model_name)
    output_df = build_output(ohlc_df, config.model.context_len, predictions, targets)
    metrics = compute_metrics(predictions[:, 0], targets)
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    export_results(
        f"{output_dir}/{config.api.curr_symb}_{start_date.strftime('%Y-%m-%d')}_{end_date.strftime('%Y-%m-%d')}",
        output_df,
        metrics,
    )
    return


if __name__ == "__main__":
    main()
