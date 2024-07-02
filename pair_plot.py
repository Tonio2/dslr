import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

import sys

def main(filename):
    df = pd.read_csv(filename)
    
    col_heads = [col_name for col_name in df.columns if df[col_name].dtypes == "float64"]
    col_heads.append('Hogwarts House')
    
    tmp = df.loc[:,col_heads]
    
    tmp = tmp.dropna()
    
    print(tmp)
    
    colors = {'Ravenclaw': 'blue', 'Slytherin': 'green', 'Gryffindor': 'red', 'Hufflepuff': 'yellow'}
    for i in tmp.index.tolist():
        value = colors[tmp.at[i, 'Hogwarts House']]
        tmp.at[i, 'Hogwarts House'] = value
    
    

    # Plotting the scatter matrix
    scatter_matrix(tmp, alpha=0.8, figsize=(18, 18), diagonal='kde', c=[color for color in tmp['Hogwarts House']])

    plt.show()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        exit("Usage: python describe.py dataset_train.csv")
    main(sys.argv[1])