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
    gryffindor_grades = [line["Flying"], line["Transfiguration"]]
    gryffindor_grades = [grade for grade in gryffindor_grades if not np.isnan(grade)]

    if len(gryffindor_grades) > 0:
        gryffindor_betas = [gryffindor_betas[i] for i in range(len(gryffindor_grades) + 1)]
        if predict(gryffindor_betas, gryffindor_grades):
            return "Gryffindor"
    if predict(betas[3:5], [line["Divination"]]):
        return "Slytherin"
    if predict(betas[5:], [line["Charms"]]):
        return "Ravenclaw"
    return "Hufflepuff"


def main(filename, betas):
    df = pd.read_csv(filename)
    df_pred = pd.DataFrame()
    df_pred["Index"] = df.index
    df["Flying"] = (df["Flying"] - df["Flying"].mean()) / df["Flying"].std()
    df["Transfiguration"] = (df["Transfiguration"] - df["Transfiguration"].mean()) / df[
        "Transfiguration"
    ].std()
    df["Divination"] = (df["Divination"] - df["Divination"].mean()) / df[
        "Divination"
    ].std()
    df["Charms"] = (df["Charms"] - df["Charms"].mean()) / df["Charms"].std()
    df_pred["Hogwarts House"] = df["Index"].map(lambda x: sort(df.loc[x], betas))

    df_pred.to_csv("houses.csv", index=False)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        exit("Usage: python logreg_train.py dataset_test.csv weights.txt")

    betas = []

    with open("weights.txt") as f:
        for _ in range(7):
            betas.append(float(f.readline()))
    main(sys.argv[1], betas)
