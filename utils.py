import warnings
import os
from datetime import timedelta, datetime
import pandas as pd
import glob
import os
import re
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)


def str_to_datetime(date_str):
    if date_str is None:
        return datetime.today()
    date_format = "%Y-%m-%d"
    return datetime.strptime(date_str, date_format)


def datetime_to_timestamp(dt):
    return int(dt.timestamp())


def generate_time_intervals(start_date, end_date, step_sec, max_res):
    intervals = []
    current_dt = start_date
    while current_dt < end_date:
        start_dt = current_dt
        end_dt = min(start_dt + timedelta(seconds=step_sec * max_res), end_date)
        intervals.append((start_dt, end_dt))
        current_dt = end_dt
    return intervals


def get_history(client, market_symbol, step_sec, limit, start_dt, end_dt):
    limit = min((end_dt - start_dt) / timedelta(seconds=step_sec), limit)
    data = client.get(
        f"ohlc/{market_symbol}/",
        {
            "step": int(step_sec),
            "limit": int(limit),
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


def read_output(base_path):

    csv_files = glob.glob(os.path.join(base_path, "**/output.csv"), recursive=True)
    dfs = []

    for file in csv_files:
        # Extract curr_symb, start_date, and end_date from the file name
        parent_folder = os.path.dirname(file)
        last_element = os.path.basename(parent_folder)
        match = re.match(r"(\w+)_(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})", last_element)
        if match:
            curr_symb = match.group(1)
            # start_date = match.group(2)
            # end_date = match.group(3)

            df = pd.read_csv(file, index_col=0)
            df.rename(
                columns={
                    "target": f"target_{curr_symb}",
                    "prediction": f"prediction_{curr_symb}",
                },
                inplace=True,
            )
            dfs.append(df)

    # Concatenate all the dataframes
    result = pd.concat(dfs, axis=1)

    return result