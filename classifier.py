

import re
import sys
import numpy as np



# Sigmoid (activation function)
def sigmoid(x):
    return 1.0/(1.0+math.exp(-x))

# Sigmoid derivative
def d_sigmoid(x):
    return sigmoid(x)*(1.0-sigmoid(x))


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
	[pre, targets] = read_input()
	print(targets)


<<<<<<< Updated upstream
if __name__ == '__main__':
    main()


=======






epochs = 1

training_set = 0.8

learning_rate = 0.05


# for epoch in range(epochs):
	# for sample in range(training_set):






# Read vectors and target classes from file
[pre, targets] = read_input()

# Print entire array
np.set_printoptions(threshold=sys.maxsize)

# Test to see the ouput is correct
print(np.append(pre, targets, axis=1))
>>>>>>> Stashed changes









