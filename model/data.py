import os
import logging
from datetime import datetime
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning import LightningDataModule
from typing import List
from api_client import BitstampClient
from utils import get_history, generate_time_intervals, str_to_datetime


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
    return torch.tensor(ohlc_df.loc[context_len:, "close"].values).unsqueeze(1)


class TimeSeriesDataModule(LightningDataModule):
    def __init__(
        self,
        client,
        curr_symbs,
        context_len,
        input_cols,
        start_date,
        end_date,
        step_sec=3600,
        max_res=1000,
        batch_size=64,
        val_size=0.2,
    ):
        super(TimeSeriesDataModule, self).__init__()

        for curr_symb in curr_symbs:
            ohlc_df = get_ohlc_df(client, start_date, end_date, step_sec, max_res, curr_symb)
            context = build_context(ohlc_df, context_len, input_cols)
            target = build_target(ohlc_df, context_len)

            if curr_symb == curr_symbs[0]:
                X = context
                y = target
            else:
                X = torch.cat((X, context), dim=1)
                y = torch.cat((y, target), dim=1)

        self.split_data(X, y, val_size=val_size)
        self.scale()
        self.batch_size = batch_size

    def split_data(self, X, y, val_size: float):
        n = len(X)
        n_val = int(n * val_size)
        self.X_train, self.X_val = X[:n_val], X[:n_val]
        self.y_train, self.y_val = y[:n_val], y[:n_val]
        logger.info(f"Data split into training ({n-n_val}) and validation ({n_val}) sets")

    def scale(self):
        self.min_X = self.X_train.min(dim=0, keepdim=True).values
        self.max_X = self.X_train.max(dim=0, keepdim=True).values
        self.min_y = self.y_train.min(dim=0, keepdim=True).values
        self.max_y = self.y_train.max(dim=0, keepdim=True).values
        self.X_train = (self.X_train - self.min_X) / (self.max_X - self.min_X)
        self.X_val = (self.X_val - self.min_X) / (self.max_X - self.min_X)
        self.y_train = (self.y_train - self.min_y) / (self.max_y - self.min_y)
        self.y_val = (self.y_val - self.min_y) / (self.max_y - self.min_y)
        return

    def setup(self, stage=None):
        self.train_dataset = TensorDataset(self.X_train, self.y_train)
        self.val_dataset = TensorDataset(self.X_val, self.y_val)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)


if __name__ == "__main__":
    curr_symbs = ["btceur", "etheur", "ltceur"]
    context_len = 24
    input_cols = ["close", "volume"]
    start_date = str_to_datetime("2021-01-01")
    end_date = str_to_datetime("2021-02-01")
    api_key = os.getenv("BITSTAMP_API_KEY")
    api_secret = os.getenv("BITSTAMP_API_SECRET")
    client = BitstampClient(api_key, api_secret)
    data = TimeSeriesDataModule(client, curr_symbs, context_len, input_cols, start_date, end_date)
    data.setup()
    print(data.X_train.shape, data.y_train.shape, data.X_val.shape, data.y_val.shape)
    elem = next(iter(data.train_dataloader()))
    print(len(elem), elem[0].shape, elem[1].shape)
    print(elem[0][0], elem[1][0])
