import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import yaml
import sys

def plot_logistic_classification(x, y, weights):
    if x.shape[1] not in [1, 2]:
        raise ValueError("Dataframe must have either one or two variables")
    
    def logistic_function(x):
        return 1 / (1 + np.exp(-x))
    
    if x.shape[1] == 1:
        x_values = np.linspace(x.min(), x.max(), 300)
        y_values = logistic_function(weights[0] + weights[1] * x_values)
        
        plt.figure(figsize=(10, 6))
        plt.plot(x_values, y_values, label='Logistic Function', color='blue')
        plt.scatter(x, y, marker='x', color='red', label='Data Points')
        plt.xlabel(df.columns[0])
        plt.ylabel('Probability')
        plt.title('Logistic Function Visualization (2D)')
        plt.legend()
        plt.show()
    
    elif x.shape[1] == 2:
        x1 = x.iloc[:, 0]
        x2 = x.iloc[:, 1]
        x1_values = np.linspace(x1.min(), x1.max(), 100)
        x2_values = np.linspace(x2.min(), x2.max(), 100)
        x1_grid, x2_grid = np.meshgrid(x1_values, x2_values)
        z_values = logistic_function(weights[0] + weights[1] * x1_grid + weights[2] * x2_grid)
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(x1_grid, x2_grid, z_values, color='blue', alpha=0.6)
        ax.scatter(x1, x2, y, color='red', marker='x', label='Data Points')
        ax.set_xlabel(x.columns[0])
        ax.set_ylabel(x.columns[1])
        ax.set_zlabel('Probability')
        ax.set_title('Logistic Function Visualization (3D)')
        plt.show()

if len(sys.argv) != 2:
        exit("Usage: python plotweights.py house")

house = sys.argv[1]

if house not in ["Gryffindor", "Slytherin", "Ravenclaw", "Hufflepuff"]:
    exit("House must be one of Gryffindor, Slytherin, Ravenclaw, or Hufflepuff")
    
with open('params.yaml', 'r') as file:
    params = yaml.safe_load(file)
    
df = pd.read_csv("datasets/dataset_train.csv")

# Map the Hogwarts House column
y = df["Hogwarts House"].map(
        {"Ravenclaw": 0, "Slytherin": 0, "Gryffindor": 0, "Hufflepuff": 0, house: 1}
    )

# Select the relevant features for Gryffindor from YAML file
features = params[house]
x = df[features].copy()

# Normalize the selected features
for col in x.columns:
    x.loc[:, col] = (x[col] - x[col].mean()) / x[col].std()

# Load weights from the file
weights = []
with open("weights.txt") as f:
    for _ in range(8):
        weights.append(float(f.readline()))

# Plot the logistic classification (assuming you have this function defined elsewhere)
weight_start = 0
for house_name in params:
    if house_name == house:
        break
    weight_start += len(params[house_name]) + 1
plot_logistic_classification(x, y, weights[weight_start:weight_start + len(params[house]) + 1])

