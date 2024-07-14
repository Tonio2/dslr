import sys
import pandas as pd

house_features = {
    "Gryffindor": ["Flying", "History of Magic"],
    "Slytherin": ["Divination", "Herbology"],
    "Ravenclaw": ["Charms"],
    "Hufflepuff": [],
}


def main(filename):
    df = pd.read_csv(filename)
    df_pred = pd.read_csv("houses.csv")

    sum = 0
    for idx in df.index:
        if df_pred.loc[idx, "Hogwarts House"] == df.loc[idx, "Hogwarts House"]:
            sum += 1
        else:
            notes_actual = [f"{feature} : {df.loc[idx, feature]}" for feature in house_features[df.loc[idx, "Hogwarts House"]]]
            notes_predicted = [f"{feature} : {df.loc[idx, feature]}" for feature in house_features[df_pred.loc[idx, "Hogwarts House"]]]
            print(
                f"Row {idx} - predicted: {df_pred.loc[idx, 'Hogwarts House']}, actual: {df.loc[idx, 'Hogwarts House']} - {notes_predicted} - {notes_actual}"
            )

    print(f"Accuracy: {sum / len(df)}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        exit("Usage: python logreg_train.py dataset_test.csv")

    main(sys.argv[1])
