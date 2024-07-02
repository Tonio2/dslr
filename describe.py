import pandas as pd
import math
import sys
import pdb

def clean(values):
    return [v for v in values if not math.isnan(v)]

def mean(values):
    total = 0.0

    for val in values:
        total += val

    return total / len(values)

def std(values):
    total = 0.0
    mean_val = mean(values)

    for v in values:
        total += (v - mean_val) * (v - mean_val)

    return math.sqrt(total / len(values))

def min(values):
    res = values[0]

    for i in range(1, len(values)):
        if values[i] < res:
            res = values[i]

    return res

def max(values):
    res = values[0]

    for i in range(1, len(values)):
        if values[i] > res:
            res = values[i]

    return res

def sort(values):
    for i in range(len(values)):
        inf = values[i]
        inf_idx = i
        for j in range(i, len(values)):
            if values[j] < inf:
                inf = values[j]
                inf_idx = j
        values[i], values[inf_idx] = values[inf_idx], values[i]
        
def quantile(values, q):
    pos = q * (len(values) + 1) - 1
    idx = int(pos)
    if idx == pos:
        return values[idx]
    
    return values[idx] + (values[idx + 1] - values[idx]) * (pos - idx)

def main(filename):
    df = pd.read_csv(filename)
    row_heads = ["Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max"]
    col_heads = [col_name for col_name in df.columns if df[col_name].dtypes == "float64"]
    res = pd.DataFrame(0.0, index=row_heads, columns=col_heads)

    for col_name in res.columns:
        values = clean(df[col_name])
        if len(values) == 0:
            continue
        res.at["Count", col_name] = len(values)
        res.at["Mean", col_name] = mean(values)
        res.at["Std", col_name] = std(values)
        res.at["Min", col_name] = min(values)
        res.at["Max", col_name] = max(values)
        sort(values)
        res.at["25%", col_name] = quantile(values, 0.25)
        res.at["50%", col_name] = quantile(values, 0.5)
        res.at["75%", col_name] = quantile(values, 0.75)

    print(res)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        exit("Usage: python describe.py dataset_train.csv")
    main(sys.argv[1])


