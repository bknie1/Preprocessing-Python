# Data Preprocessing Template

# Notes:
# Missing Data - Use the mean from other columns to make an estimate.

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# For ML Models, Preprocessing Classes, Imputer Class for Missing Data
from sklearn.preprocessing import Imputer

# Importing the dataset
dataset = pd.read_csv('Data.csv')
print("Dataset:\n", dataset)
# The Matrix of Features
# Take all lines, all columns except last! (-1)
x = dataset.iloc[:, :-1].values
print("Independent:\n", x)

# The Dependent Variable Vector
# Take all lines, and only third column.
y = dataset.iloc[:, 3].values
print("Dependent:\n", y)


# Taking Care of Missing Data
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
# Fit imputer to Matrix 'x'
imputer = imputer.fit(x[:, 1:3])  # 1-2: Upperbound is excluded.

x[:, 1: 3] = imputer.transform(x[:, 1: 3])

print(x)