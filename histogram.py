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

def main(filename):
    pdb.set_trace()
    df = pd.read_csv(filename)
    
    house_names = get_unique_values(df['Hogwarts House'])
    col_heads = [col_name for col_name in df.columns if df[col_name].dtypes == "float64"]
    res = pd.DataFrame(0.0, index=house_names, columns=col_heads)
    count = pd.DataFrame(0, index=house_names, columns=col_heads)
    
    for col_name in col_heads:
        for i in range(len(df[col_name])):
            if not pd.isnull(df[col_name][i]):
                res.at[df['Hogwarts House'][i], col_name] += df[col_name][i]
                count.at[df['Hogwarts House'][i],col_name] += 1
        for house in house_names:
            res.at[house, col_name] = res.at[house, col_name] / count.at[house, col_name]
    
    fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(15, 25))
    axes = axes.flatten()
    
    colors = ['blue', 'green', 'red', 'yellow']


    for i, column in enumerate(res.columns):
        res[column].plot(kind='bar', ax=axes[i], title=column, color=colors)
        axes[i].set_xlabel('')
        axes[i].set_ylabel('Value')
        axes[i].set_xticklabels([])  # Remove x-axis labels

        
    from matplotlib.patches import Patch

    legend_elements = [Patch(facecolor=colors[i], label=house) for i, house in enumerate(house_names)]
    fig.legend(handles=legend_elements, loc='upper center', ncol=4, fontsize='large', title='Houses')


    plt.tight_layout(rect=[15.0, 0.03, 1, 0.95])
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        exit("Usage: python describe.py dataset_train.csv")
    main(sys.argv[1])