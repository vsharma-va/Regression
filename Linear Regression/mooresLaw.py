import numpy as np
import matplotlib.pyplot as plt
import re
import pandas as pd
import math

nonDecimal = re.compile(r'[^\d]+')

X = []
Y = []
Z = []
K = []


for line in open("../Resources/moore.csv"):
    r = line.split('\t')
    x = nonDecimal.sub('', r[2].split('[')[0])
    y = nonDecimal.sub('', r[1].split('[')[0])
    X.append([float(x), float(1)])
    Y.append([float(y)])
    Z.append(int(x))
    K.append(float(y))

X = np.array(X)
Y = np.array(Y)
# for normalization i started with mean and then increased values so that i donot get errors while taking log
X[:, 0] = (X[:, 0] - 1999.666)/2016
plt.scatter(X[:, 0], Y)
plt.show()

linearY = np.log(Y).reshape(len(Y), 1)
thetas = np.zeros((2, 1), dtype=float)
plt.scatter(X[:, 0], linearY)
plt.show()


def calculateCost(x, y, theta):
    m = len(x)
    prediction = np.dot(x, theta)
    squaredError = (prediction - y) ** 2
    costFunction = 1/(2*m) * np.sum(squaredError)
    return costFunction


def gradientDescent(x, y, theta, alpha, iterations):
    m = len(x)
    for i in range(iterations):
        prediction = np.dot(x, theta)
        error = np.dot(x.transpose(), (prediction - y))
        descent = alpha * 1/m * error
        theta -= descent
    return theta


cost = calculateCost(X, linearY, thetas)
thetas = gradientDescent(X, linearY, thetas, 1.9, 310000)
equation = f'{thetas[1, :][0]} + {thetas[0, :][0]} * X'
print('Equation is: ', '\n', equation)
print('Cost Function: ', cost)
yCap = np.dot(X, thetas)

plt.plot(X[:, 0], yCap, label='Linear Regression', color='r')
plt.scatter(X[:, 0], linearY, label='data')
plt.xlabel('Years')
plt.ylabel('Transistor Count')
plt.legend(loc='best')
plt.title('Moore\'s law')
plt.show()


def predict(year):
    transistors = thetas[1, :] + thetas[0, :] * year
    return transistors[0]


predict1 = predict(2016)
print(predict1) # predict doesnt work
