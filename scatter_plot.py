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

    feat1 = 'Astronomy'
    feat2 = 'Defense Against the Dark Arts'
    df = pd.read_csv(filename)
    
    house_names = get_unique_values(df['Hogwarts House'])
    
    # Create a pair plot for feat1 and feat2 with a color for each house
    fig, ax = plt.subplots()
    colors = ['blue', 'green', 'red', 'yellow']
    for i, house in enumerate(house_names):
        df_house = df[df['Hogwarts House'] == house]
        ax.scatter(df_house[feat1], df_house[feat2], c=colors[i], label=house)

    # Set the legend out of the graph to improve readibility
    ax.legend(title='Houses', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_xlabel(feat1)
    ax.set_ylabel(feat2)
    
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        exit("Usage: python describe.py dataset_train.csv")
    main(sys.argv[1])