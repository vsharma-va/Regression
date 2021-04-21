import numpy as np
from matplotlib import pyplot as plt
import re

pattern = re.compile(r"[^\d]")

X = []
Y = []

for line in open("../Resources/moore.csv"):
    r = line.split('\t')
    # this is splitting at the first square brackets and returning the first value
    y = re.sub(pattern, '', r[1].split('[')[0])
    x = re.sub(pattern, '', r[2].split('[')[0])

    X.append(int(x))
    Y.append(int(y))

X = np.array(X)
Y = np.array(Y)

plt.scatter(X, Y)
plt.show()

# since it is an exponential function we will make it linear by taking log
linearY = np.log(Y)

plt.scatter(X, linearY)
plt.show()

# using the same formula calculated before
denominator = X.dot(X) - X.mean() * X.sum()
a = (X.dot(linearY) - linearY.mean() * X.sum()) / denominator
b = (linearY.mean() * X.dot(X) - X.mean() * X.dot(linearY)) / denominator

YCap = a*X + b

print('a: ', a, 'b: ', b)

plt.scatter(X, linearY)
plt.plot(X, YCap, color='r')
plt.show()

# Accuracy of the model
d1 = linearY - YCap
d2 = linearY - linearY.mean()
rSquared = 1 - (d1.dot(d1) / d2.dot(d2))
print("the r-squared is: ", rSquared)

# showing amount doubles every two years
print("Time to double is: ", np.log(2)/a, "years.")
