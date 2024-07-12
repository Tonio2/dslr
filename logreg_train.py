import sys
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np


class LogisticRegression:
    def __init__(self, x, y, alpha=0.1, n_iter=10000):
        self.alpha = alpha
        self.n_iter = n_iter
        self.x = np.array(x)
        self.y = np.array(y)
        self.betas = np.zeros(self.x.shape[1] + 1)
        self.convergence_data = []
        self.x = np.c_[np.ones(self.x.shape[0]), self.x]  # Add intercept term

    def fit(self):
        m = len(self.x)
        for _ in range(self.n_iter):
            y_pred = self.hypothesis(self.x)
            error = y_pred - self.y
            deltas = np.dot(self.x.T, error) / m
            self.betas -= self.alpha * deltas
            L = -np.sum(self.y * np.log(y_pred) + (1 - self.y) * np.log(1 - y_pred)) / m
            self.convergence_data.append(L)

    def hypothesis(self, x):
        return 1 / (1 + np.exp(-np.dot(x, self.betas)))

    def predict(self, x):
        x = np.c_[np.ones(x.shape[0]), x]  # Add intercept term
        return np.where(self.hypothesis(x) >= 0.5, 1, 0)

    def get_params(self):
        return self.betas
        
    def plot_convergence_data(self):
        df = pd.DataFrame(self.convergence_data, columns=['Values'])

        # Plot the DataFrame
        plt.figure(figsize=(10, 6))
        df['Values'].plot(kind='line', marker='o')
        plt.title('Plot of Values List')
        plt.xlabel('Index')
        plt.ylabel('Values')
        plt.grid(True)
        plt.show()


def format_data(df, col_names, house):
    data = df.dropna(subset=col_names)
    
    y = data["Hogwarts House"].map(
        {"Ravenclaw": 0, "Slytherin": 0, "Gryffindor": 0, "Hufflepuff": 0, house: 1}
    )
    x = data[col_names]
    for col in col_names:
        x[col] = (x[col] - x[col].mean()) / x[col].std()
    return x.values.tolist(), y.values.tolist()


def main(filename):
    df = pd.read_csv(filename)
    with open("weights.txt", "w") as f:
        x, y = format_data(df, ["Flying", "Transfiguration"], "Gryffindor")
        logreg = LogisticRegression(x, y)
        logreg.fit()
        f.write(f"{logreg.get_params()[0]}\n{logreg.get_params()[1]}\n{logreg.get_params()[2]}\n")

        x, y = format_data(df, ["Divination", "Astronomy"], "Slytherin")
        logreg2 = LogisticRegression(x, y)
        logreg2.fit()
        f.write(f"{logreg2.get_params()[0]}\n{logreg2.get_params()[1]}\n{logreg2.get_params()[2]}\n")

        x, y = format_data(df, ["Charms"], "Ravenclaw")
        logreg3 = LogisticRegression(x, y)
        logreg3.fit()
        f.write(f"{logreg3.get_params()[0]}\n{logreg3.get_params()[1]}")
        
        logreg.plot_convergence_data()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        exit("Usage: python logreg_train.py dataset_train.csv")
    main(sys.argv[1])
