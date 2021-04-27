import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("../Resources/multivariate.csv", header=None)

np.seterr(all='warn')
X2 = data.iloc[:, 0:2]
Y = data[2]
X2 = np.array(X2)
Y = np.array(Y).reshape(len(Y), 1)
m = len(X2)


def normalizeFeatures(x):
    mean = np.mean(x, axis=0)  # in numpy axis=0 is along the columns
    std = np.std(x, axis=0)
    X_norm = (x - mean) / std
    return X_norm


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
equation = f'{thetas[2, :][0]} + {thetas[1, :][0]} * x1 + {thetas[0, :][0]} * x2'
print(equation)
yCap = np.dot(X2, thetas)

# rSquared for accuracy of our model
predicted = yCap.reshape(len(yCap),)
actual = Y.reshape(len(Y),)
d1 = actual - predicted
d2 = actual - actual.mean()
rSquared = 1 - d1.dot(d1)/d2.dot(d2)
print('rSquared is:', rSquared)

# graph of cost function
# the cost Function decreases with correct learning rate
# if the learning rate is very high then it will miss the global minimum and keep increasing to infinity
# that is why we get the error invalid value when subtracting thetas and descent
plt.plot(costHistory)
plt.show()


# prediction
def prediction(size, numberBedrooms):
    featureArray = np.array([size, numberBedrooms, 1]).reshape(1, 3)
    costOfHouse = np.dot(featureArray, thetas)
    return costOfHouse[0][0]


normSampleData = normalizeFeatures(np.array([1650, 3]))
print(prediction(normSampleData[0], normSampleData[1]))

