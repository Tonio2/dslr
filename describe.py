import pandas as pd
from utils import *
import sys

import numpy as np


def main(filename):
    df = pd.read_csv(filename)
    row_heads = ["Count", "Mean", "Variance", "Std", "Min", "25%", "50%", "75%", "Max", "Range", "Missing values"]
    col_heads = [
        col_name for col_name in df.columns if df[col_name].dtypes == "float64"
    ]
    res = pd.DataFrame(0.0, index=row_heads, columns=col_heads)

    for col_name in res.columns:
        values = clean(df[col_name])
        if len(values) == 0:
            continue
        res.at["Range", col_name] = max_val(values) - min_val(values)
        res.at["Missing values", col_name] = missing_val(df[col_name])
        res.at["Count", col_name] = len(values)
        res.at["Mean", col_name] = mean(values)
        res.at["Variance", col_name] = variance(values)
        res.at["Std", col_name] = std(values)
        res.at["Min", col_name] = min_val(values)
        res.at["Max", col_name] = max_val(values)
        sort(values)
        res.at["25%", col_name] = quantile(values, 0.25)
        res.at["50%", col_name] = quantile(values, 0.5)
        res.at["75%", col_name] = quantile(values, 0.75)
        res.at["Range", col_name] = max_val(values) - min_val(values)

    print(res)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        exit("Usage: python describe.py dataset_train.csv")
    main(sys.argv[1])
