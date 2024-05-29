from utils import read_output


def apply_strategy(base_path: str, curr_symbs, fees: float = 0.003, cash: float = 100.0) -> None:
    df = read_output(base_path).reset_index()
    for i, row in df.iterrows():
        if i == 0:
            df.at[i, "cash"] = cash
            df.at[i, "curr_pred"] = None
            df.at[i, "perf_pred"] = None
            df.at[i, "perf_realized"] = None
            continue
        perfs = [
            row[f"target_{curr_symb}"] / df.iloc[i - 1][f"target_{curr_symb}"] - 1
            for curr_symb in curr_symbs
        ]
        perfs_pred = [
            row[f"prediction_{curr_symb}"] / df.iloc[i - 1][f"target_{curr_symb}"] - 1
            for curr_symb in curr_symbs
        ]
        best_perf_pred = max(perfs_pred)
        max_index = perfs_pred.index(best_perf_pred)
        best_curr_symb_pred = curr_symbs[max_index]
        perf_realized = perfs[max_index]
        cash *= 1 + perf_realized - fees
        cash = round(max(0, cash), 5)
        df.at[i, "cash"] = cash
        df.at[i, "curr_pred"] = best_curr_symb_pred
        df.at[i, "perf_pred"] = best_perf_pred
        df.at[i, "perf_realized"] = perf_realized
        df.to_csv(f"{base_path}/strategy_output.csv")
    return