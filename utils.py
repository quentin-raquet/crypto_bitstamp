import warnings
from datetime import timedelta
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

def datetime_to_timestamp(dt):
    return int(dt.timestamp())


def generate_time_intervals(start_date, end_date, step_sec, max_res):
    intervals = []
    current_dt = start_date
    while current_dt < end_date:
        start_dt = current_dt
        end_dt = start_dt + timedelta(seconds=step_sec * max_res)
        intervals.append((start_dt, end_dt))
        current_dt = end_dt
    return intervals


def get_history(client, market_symbol, step_sec, start_dt, end_dt):
    data = client.get(
        f"ohlc/{market_symbol}/",
        {
            "step": int(step_sec),
            "limit": 1000,
            "start": datetime_to_timestamp(start_dt),
            "end": datetime_to_timestamp(end_dt),
        },
    )["data"]["ohlc"]
    df = pd.DataFrame(data)
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
    df = df.astype({"close": "float64", "volume": "float64"})
    df.rename(
        columns={"close": f"close", "volume": f"volume"},
        inplace=True,
    )
    return df
