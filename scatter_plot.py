import pandas as pd
import matplotlib.pyplot as plt
import sys
import pdb

def get_unique_values(values):
    res = []
    
    for v in values:
        if v not in res:
            res.append(v)
    
    return res

def display(df):
    data = []
    
    for col in df:
        for v in df[col]:
            data.append([v, col])
    
    new_df = pd.DataFrame(data)
    
    new_df.plot.scatter(1,0)

def main(filename):

    df = pd.read_csv(filename)
    
    house_names = get_unique_values(df['Hogwarts House'])
    col_heads = [col_name for col_name in df.columns if df[col_name].dtypes == "float64"]
    
    fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(15, 25))
    axes = axes.flatten()
    
    for i, col in enumerate(col_heads):
        df.plot(kind='scatter', x='Hogwarts House', y=col, ax=axes[i])
    
    plt.tight_layout(rect=[15.0, 0.03, 1, 0.95])
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        exit("Usage: python describe.py dataset_train.csv")
    main(sys.argv[1])