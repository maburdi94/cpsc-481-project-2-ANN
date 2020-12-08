
import utils
import re
import sys
import numpy as np
import math


# Sigmoid (activation function)
sigmoid = np.vectorize(lambda x : 1.0/(1.0+math.exp(-x)))


# Read classified set and target values from input file
def read_input(filename = "input.txt"):
    file 	 = open(filename, "r")

    # Load the pre-classified vectors from file and the targets
    pre 	 = np.empty((0,10))
    targets  = np.empty((0,1))

    for line in file:
        group     = re.search("\( *(\d+) *\(((?:\d+ *)+) *\) *(\d+)\)", line).groups()

        _id  	  = int(group[0])
        _features = [int(i) for i in group[1].split(' ')]
        _class 	  = int(group[2])

        pre = np.append(pre, [_features], axis=0)
        targets = np.append(targets, [[_class]], axis=0)

    file.close()

    return [pre, targets]






def main():
    epochs = 5
    LR = 0.01

    #training_set = 0.8
    #learning_rate = 0.05

    # for epoch in range(epochs):
    # for sample in range(training_set):
    # Read vectors and target classes from file
    [pre, expected] = read_input()

    # Print entire array
    np.set_printoptions(threshold=sys.maxsize)
    np.set_printoptions(linewidth=1000)


    # Get a vector of 4 weights initialized with random weights
    w1 = np.random.uniform(-1, 1, (4, 10))
    w2 = np.random.uniform(-1, 1, (8, 4))

    # print("Weights 1: \n", w1)
    # print("Weights 2: \n", w2)

    # 80% of inputs is training set
    training_set = pre[0:math.floor(.8*len(pre)),:]
    # print("Training Set\n", training_set)

    # 20% is holdout set
    holdout_set = pre[math.floor(.8*len(pre)):,:]
    # print("Holdout Set\n", holdout_set)

    v = pre[0:1,:].reshape(10,1)

    print("dim(sample[0]) = ", v.shape)
    print("dim(w1) = ", w1.shape)
    print("dim(w2) = ", w2.shape)

    print("sample[0] = ", v)
    print("w1*sample[0] = ", np.dot(w1, v))
    print("sigmoid(w1*sample[0]) = ", sigmoid(np.dot(w1, v)))

    # Training
    for epoch in range(epochs):
        print("Epoch ", epoch)
        for i, x in enumerate(pre):

            x = np.transpose([x])

            # [z] = activation([w1] x [x])
            z = sigmoid( np.dot(w1, x )) # [4 x 1]

            # [y] = activation([w2] x [z])
            y = sigmoid( np.dot(w2, z)) # [8 x 1]

            # print("dim(x) = ", np.shape(x))
            # print("dim(z) = ", np.shape(z))
            # print("dim(y) = ", np.shape(y))

            # Create a target vector [.2 .2 .2 ... .8 ... .2 .2]
            #  where .2 signifies 0 and .8 signifies 1
            target = np.full((8, 1), .2)
            target[int(expected[i])] = .8

            # print("dim(target) = ", np.shape(target))

            # Find error between guess and known class
            error = y - target

            delta1 = error * (y * (1 - y)) # [8 x 1]

            # print("dim(delta1) = ", np.shape(delta1))

            # Adjust H-to-O weights
            w2adj = LR*np.dot(delta1,np.transpose(z))
            w2 = w2 +  w2adj# [8 x 4]

            delta2 = (z * (1 - z)) * np.dot(np.transpose(w2),delta1) # [4 x 1]

            w1adj = LR*np.dot(delta2,np.transpose(x))
            w1 = w1 +  w1adj# [4 x 10]


            # Check if less than threshold
            # np.mean([w1adj, w2adj])


    # Testing
    for i, x in enumerate(pre):

        # [z] = activation([w1] x [x])
        z = sigmoid( np.dot(w1, x )) # [4 x 1]

        # [y] = activation([w2] x [z])
        y = sigmoid( np.dot(w2, z)) # [8 x 1]

        print("Target:   ", int(expected[i][0]))
        print("Sample:   ", x)
        print("H input:  ", z)
        print("H output: ", y)
        print("Classed:  ", (np.argmax(y) + 1))
        print()

    # Test to see the ouput is correct
    # print(pre)


if __name__ == '__main__':
    main()
