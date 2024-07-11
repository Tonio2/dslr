import sys
import pandas as pd


def main(filename):
    df = pd.read_csv(filename)
    df_pred = pd.read_csv("houses.csv")
    
    df_pred.set_index("Index", inplace=True)

    sum = 0
    total = 0
    for idx in df_pred.index:
        total += 1
        if df_pred.loc[idx, "Hogwarts House"] == df.loc[idx, "Hogwarts House"]:
            sum += 1
        else:
            print(
                f"Row {idx} - predicted: {df_pred.loc[idx, 'Hogwarts House']}, actual: {df.loc[idx, 'Hogwarts House']}"
            )

    print(f"Accuracy: {sum / total}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        exit("Usage: python logreg_train.py dataset_test.csv")

    main(sys.argv[1])
