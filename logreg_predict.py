import sys
import math
import pandas as pd


def hypothesis(beta0, beta1, x):
    return 1 / (1 + math.exp(-(beta0 + beta1 * x)))


def predict(beta0, beta1, x):
    return hypothesis(beta0, beta1, x) >= 0.5


def sort(line, betas):
    if predict(betas[0], betas[1], line["Flying"]):
        return "Gryffindor"
    if predict(betas[2], betas[3], line["Divination"]):
        return "Slytherin"
    if predict(betas[4], betas[5], line["Charms"]):
        return "Ravenclaw"
    return "Hufflepuff"


def main(filename, betas):
    df = pd.read_csv(filename).dropna(subset=["Flying", "Divination", "Charms"])
    df_pred = pd.DataFrame()
    df_pred["Index"] = df.index
    df["Flying"] = (df["Flying"] - df["Flying"].mean()) / df["Flying"].std()
    df["Divination"] = (df["Divination"] - df["Divination"].mean()) / df[
        "Divination"
    ].std()
    df["Charms"] = (df["Charms"] - df["Charms"].mean()) / df["Charms"].std()
    df_pred["Hogwarts House"] = df_pred["Index"].apply(lambda idx: sort(df.loc[idx], betas))

    df_pred.to_csv("houses.csv", index=False)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        exit("Usage: python logreg_train.py dataset_test.csv weights.txt")

    betas = []

    with open("weights.txt") as f:
        for _ in range(6):
            betas.append(float(f.readline()))
    main(sys.argv[1], betas)
