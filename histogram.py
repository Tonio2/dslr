import pandas as pd
import matplotlib.pyplot as plt
import sys
import numpy as np

def get_unique_values(values):
    res = []
    
    for v in values:
        if v not in res:
            res.append(v)
    
    return res

def parse_colum(df, col_name, house_names):
    ranges = np.linspace(df[col_name].min(), df[col_name].max(), 10)
    ret = pd.DataFrame(0, index=ranges, columns=house_names)
    
    for v in df.index:
        for r in ranges:
            if df[col_name][v] <= r:
                ret.at[r, df['Hogwarts House'][v]] += 1
                break
    
    return ret
    

def main(filename):
    df = pd.read_csv(filename)
    
    house_names = get_unique_values(df['Hogwarts House'])
    col_heads = [col_name for col_name in df.columns if df[col_name].dtypes == "float64"]
    
    fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(15, 25))
    axes = axes.flatten()
    
    colors = ['blue', 'green', 'red', 'yellow']

    for i, column in enumerate(col_heads):
        data = parse_colum(df, column, house_names)
        ax=axes[i]
        data.plot(kind='bar', ax=ax, title=column, color=colors, legend=False)
        ax.set_xticklabels([int(label) for label in data.index], rotation=45, ha='right')


        
    from matplotlib.patches import Patch

    legend_elements = [Patch(facecolor=colors[i], label=house) for i, house in enumerate(house_names)]
    fig.legend(handles=legend_elements, loc='upper center', ncol=4, fontsize='large', title='Houses')


    plt.tight_layout(rect=[10.0, 0.03, 1, 0.95])
    plt.subplots_adjust(hspace=0.7)
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        exit("Usage: python describe.py dataset_train.csv")
    main(sys.argv[1])