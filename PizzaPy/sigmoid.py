import numpy as np

def sigmoid(z):
  return 1 / (1 + np.exp(-z))

x = sigmoid(2)
print(x)
# np.exp uses Euler's number as the base then uses input as the exponent (2.7182818)
print(np.exp(2))