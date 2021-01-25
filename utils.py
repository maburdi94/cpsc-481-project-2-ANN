

# Sigmoid (activation function)
def sigmoid(x):
    return 1.0/(1.0+math.exp(-x))

# Sigmoid derivative
def d_sigmoid(x):
    return sigmoid(x)*(1.0-sigmoid(x))