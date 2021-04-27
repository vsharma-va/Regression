import pandas as pd
import numpy as np

data = pd.read_csv("../Resources/multivariate.csv", header=None)

np.seterr(all='warn')
X2 = data.iloc[:, 0:2]
Y = data[2]
X2 = np.array(X2)
Y = np.array(Y).reshape(len(Y), 1)
m = len(X2)


def normalizeFeatures(x):
    firstFeature = x[:, 0]
    secondFeature = x[:, 1]
    firstFeature = (firstFeature - firstFeature.mean()) / firstFeature.std()
    secondFeature = (secondFeature - secondFeature.mean()) / secondFeature.std()
    x = np.append(firstFeature.reshape(m, 1), secondFeature.reshape(m, 1), axis=1)
    return x


X2 = normalizeFeatures(X2)
# since there are two features there will be three thetas as the intercept of the two features will be added together
thetas = np.zeros((3, 1))
X2 = np.append(X2, np.ones((m, 1)), axis=1)


def computeCost(x, y, theta):
    prediction = np.dot(x, theta)
    squaredErrors = (prediction - y) ** 2
    costFunc = 1 / (2 * m) * np.sum(squaredErrors)
    return costFunc


def gradientDescent(x, y, theta, alpha, iterations):
    costHistory = []
    for i in range(iterations):
        prediction = np.dot(x, theta)
        error = np.dot(x.transpose(), (prediction - y))
        descent = alpha * 1/m * error
        theta -= descent
        costHistory.append(computeCost(x, y, theta))
    return theta, costHistory


cost = computeCost(X2, Y, thetas)
# learning rate and iterations found by trial and error
# ideal value of learning rate is less than 1 and greater than 10^-6
thetas, costHistory = gradientDescent(X2, Y, thetas, 0.01, 1000)
equation = f'{thetas[0, :][0]} + {thetas[1, :][0]} * x1 + {thetas[2, :][0]} * x2'
print(equation)
yCap = np.dot(X2, thetas)

# rSquared for accuracy of our model
predicted = yCap.reshape(len(yCap),)
actual = Y.reshape(len(Y),)
d1 = actual - predicted
d2 = actual - actual.mean()
rSquared = 1 - d1.dot(d1)/d2.dot(d2)
print('rSquared is:', rSquared)
