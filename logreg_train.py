import sys
import pandas as pd
import math


class LogisticRegression:
    def __init__(self, alpha=0.1, n_iter=10000):
        self.alpha = alpha
        self.n_iter = n_iter
        self.beta0 = 0
        self.beta1 = 0

    def fit(self, x, y):
        for _ in range(self.n_iter):
            delta0 = 0
            delta1 = 0
            for i in range(len(x)):
                delta0 += self.sigmoid(x[i]) - y[i]
                delta1 += (self.sigmoid(x[i]) - y[i]) * x[i]
            self.beta0 -= self.alpha * delta0 / len(x)
            self.beta1 -= self.alpha * delta1 / len(x)

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-(self.beta0 + self.beta1 * x)))

    def predict(self, x):
        return self.sigmoid(x).map(lambda x: 1 if x >= 0.5 else 0)

    def get_params(self):
        return [self.beta0, self.beta1]

    def set_params(self, beta0, beta1):
        self.beta0 = beta0
        self.beta1 = beta1


def format_data(df, col_name, house):
    data = df.dropna(subset=[col_name])
    x = data[col_name]
    y = data["Hogwarts House"].map(
        {"Ravenclaw": 0, "Slytherin": 0, "Gryffindor": 0, "Hufflepuff": 0, house: 1}
    )
    x = (x - x.mean()) / x.std()
    return x.values.tolist(), y.values.tolist()


def main(filename):
    df = pd.read_csv(filename)
    with open("weights.txt", "w") as f:
        logreg = LogisticRegression()

        x, y = format_data(df, "Flying", "Gryffindor")
        logreg.fit(x, y)
        f.write(f"{logreg.get_params()[0]}\n{logreg.get_params()[1]}\n")

        logreg.set_params(0, 0)
        x, y = format_data(df, "Divination", "Slytherin")
        logreg.fit(x, y)
        f.write(f"{logreg.get_params()[0]}\n{logreg.get_params()[1]}\n")

        logreg.set_params(0, 0)
        x, y = format_data(df, "Charms", "Ravenclaw")
        logreg.fit(x, y)
        f.write(f"{logreg.get_params()[0]}\n{logreg.get_params()[1]}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        exit("Usage: python logreg_train.py dataset_train.csv")
    main(sys.argv[1])
