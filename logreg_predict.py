import sys
import math
import pandas as pd
import numpy as np

def hypothesis(betas, x):
    x.insert(0, 1)
    return 1 / (1 + np.exp(-np.dot(x, betas)))

def predict(betas, x):
    return hypothesis(betas, x) >= 0.5

def sort(line, betas):
    gryffindor_betas = betas[:3]
    gryffindor_grades = [line["Flying"], line["History of Magic"]]
    gryffindor_grades = [grade for grade in gryffindor_grades if not np.isnan(grade)]

    if len(gryffindor_grades) > 0:
        gryffindor_betas = [gryffindor_betas[i] for i in range(len(gryffindor_grades) + 1)]
        if predict(gryffindor_betas, gryffindor_grades):
            return "Gryffindor"

    if predict(betas[6:], [line["Charms"]]):
        return "Ravenclaw"

    slytherin_betas = betas[3:6]
    slytherin_grades = [line["Divination"], line["Herbology"]]
    slytherin_grades = [grade for grade in slytherin_grades if not np.isnan(grade)]

    if len(slytherin_grades) > 0:
        slytherin_betas = [slytherin_betas[i] for i in range(len(slytherin_grades) + 1)]
        if predict(slytherin_betas, slytherin_grades):
            return "Slytherin"

    return "Hufflepuff"


def main(filename, betas):
    df = pd.read_csv(filename)
    df_pred = pd.DataFrame()
    df_pred["Index"] = df.index

    all_features = ["Flying", "History of Magic", "Divination", "Herbology", "Charms"]
    for feature in all_features:
        df[feature] = (df[feature] - df[feature].mean()) / df[feature].std()

    df_pred["Hogwarts House"] = df["Index"].map(lambda x: sort(df.loc[x], betas))


    df_pred.to_csv("houses.csv", index=False)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        exit("Usage: python logreg_train.py dataset_test.csv weights.txt")

    betas = []

    with open("weights.txt") as f:
        for _ in range(8):
            betas.append(float(f.readline()))
    main(sys.argv[1], betas)
