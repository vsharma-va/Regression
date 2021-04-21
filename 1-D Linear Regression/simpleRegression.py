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
YCap = a*X + b

plt.scatter(X, Y)
plt.plot(X, YCap, color = 'r')
plt.show()

# Determining how good the model is using r squared method
# method 1 (mine)
d1 = (Y - YCap)**2
SSres = d1.sum()

d2 = (Y - Y.mean())**2
SStotal = d2.sum()

rSquared = 1 - (SSres / SStotal)

print(rSquared)

# method 2 (lecture)
d1 = (Y - YCap)
d2 = (Y - Y.mean())

# since x.dot(x) = sigma(x ** 2)
rSquared = 1 - d1.dot(d1) / d2.dot(d2)

print("The r-squared is: ", rSquared)
