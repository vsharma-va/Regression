import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("../Resources/data_1d.csv", header=None)
df = pd.DataFrame(data)

df.insert(1, 'extra', np.ones((100, 1)), True)
X = df.iloc[:, 0:2]
Y = df.iloc[:, 2]
theta = np.zeros((2, 1), dtype=float)
X = np.array(X, dtype=float)
print(X.shape)
Y = np.array(Y, dtype=float).reshape(100, 1)
print(Y.shape)
plt.scatter(X[:, 0], Y)
plt.show()


def calculateCost(x, y, theta):
    m = len(x)
    prediction = np.dot(x, theta)
    squaredError = (prediction - y) ** 2
    return 1/(2*m) * np.sum(squaredError)


def gradientDescent(x, y, theta, alpha, iterations):
    m = len(x)
    for i in range(iterations):
        prediction = x.dot(theta)
        error = np.dot(x.transpose(), (prediction - y))
        descent = alpha * 1/m * error
        theta -= descent
    return theta


cost = calculateCost(X, Y, theta)
theta = gradientDescent(X, Y, theta, 0.000001, 3000)
equation = f'{theta[1, :]} + {theta[0, :]} * X'
print(cost)
print(theta)
print(equation)
yCap = X.dot(theta)
plt.plot(X[:, 0], yCap, color='r', label='linear regression')
plt.scatter(X[:, 0], Y, label='data')
plt.legend(loc='best')
plt.show()
