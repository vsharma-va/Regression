import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("../Resources/foodData.csv", header=None)
df = pd.DataFrame(data)

# profit in 10000 dollars
# population also in 10000


def calculateCost(X, Y, thetas):
    m = len(X)
    prediction = np.dot(X, thetas)
    squaredError = (prediction - Y) ** 2
    costFunction = 1 / (2 * m) * np.sum(squaredError)
    return costFunction


# inserting a column of ones to easily calculate prediction without worrying about two thetas
# eg [12.3, 1] * [1, 2nd row is the y intercept and 1st row is the slope
#                 2]
# => 12.3 * 1 + 1 * 2 --- it gives us the same equation of straight line
df.insert(1, 'extra', np.ones((97, 1)), True)
X = df.iloc[:, 0:2]
m = len(X)
Y = df.iloc[:, 2]
X = np.array(X)
Y = np.array(Y).reshape(m, 1)
thetas = np.zeros((2, 1), dtype=float)
learningRate = 0.01
calculateCost(X, Y, thetas)  # costFunction or the squaredError


def computeGradientDescent(X, Y, theta, alpha, iterations):
    m = len(Y)
    for i in range(iterations):
        prediction = X.dot(theta)
# here we are taking the complete X matrix instead of only the first column because when you differentiate the cost
# function to minimize it the differentiation of hypothesis X * theta turns out to be only X where X is a matrix
# with a column of ones
        error = np.dot(X.transpose(), (prediction - Y))
        descent = alpha * 1/m * error
        theta -= descent
    return theta


thetas = computeGradientDescent(X, Y, thetas, learningRate, 1500)
print('h(x) =', thetas[1, :][0], '+', thetas[0, :][0], '* X')  # equation of our predicted line with thetas optimized
# calculating our predicted y
yCap = np.dot(X, thetas)
plt.plot(X[:, 0], yCap, color='r', label='linear regression')
plt.scatter(X[:, 0], Y, label='data')
plt.xlabel('Population of City')
plt.ylabel('Profit of food truck')
plt.title('Linear regression univariate')
plt.legend(loc='best')
plt.show()


def predictInMatrices(population, theta):
    # input in matrices
    profit = np.dot(population, theta)
    return profit


def predictInValues(population):
    profit = thetas[1, :][0] + thetas[0, :][0] * (population/10000)
    return profit * 10000


predict1 = predictInMatrices(np.array([3.5, 1]), thetas) * 10000
predict2 = predictInValues(35000)
print(predict1[0])
print(predict2)
