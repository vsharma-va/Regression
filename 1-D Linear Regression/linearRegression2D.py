import numpy as np
from mpl_toolkits.mplot3d import axes3d
from matplotlib import pyplot as plt

# loading the data

X = []
Y = []

for line in open("../Resources/data_2d.csv"):
    x1, x2, y = line.split(',')
# in the notebook we convert the y intercept into w by multiplying it with x which is always equal to 1
# to simulate that we add a column of 1s in X
    X.append([float(x1), float(x2), 1])
    Y.append(float(y))

# converting X and Y to arrays
X = np.array(X)
Y = np.array(Y)

W = np.linalg.solve(X.transpose() * X, X.transpose() * Y)
