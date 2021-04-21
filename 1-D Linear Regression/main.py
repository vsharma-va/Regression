import numpy as np
from matplotlib import pyplot as plt

X = []
Y = []

for line in open("../resources/data_1d.csv"):
    x, y = line.split(',')
    X.append(x)
    Y.append(y)

X = np.array(X, dtype=float)
Y = np.array(Y, dtype=float)

plt.scatter(X, Y)
plt.show()

denominator = X.dot(X) - X.mean() * X.sum()
a = (X.dot(Y) - Y.mean() * X.sum()) / denominator
b = (Y.mean() * X.dot(X) - X.mean() * X.dot(Y)) / denominator

# predicted y
yhat = a*X + b

plt.scatter(X, Y)
plt.plot(X, yhat, color = 'r')
plt.show()

