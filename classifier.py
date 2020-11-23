

import re
import sys
import numpy as np
import utils
import math


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

	epochs = 1

	# Learning Rate
	LR = 0.01

	# Read vectors and target classes from file
	[pre, targets] = read_input()

	# NumPy print options
	np.set_printoptions(threshold=sys.maxsize)


	# Get a vector of 4 weights initialized with random weights
	weights = np.random.uniform(-1, 1, (4, 10))
	print("Weights\n", weights)

	# 80% of inputs is training set
	training_set = pre[0:math.floor(.8*len(pre)),:]
	print("Training Set\n", training_set)

	# 20% is holdout set
	holdout_set = pre[math.floor(.8*len(pre)):,:]
	print("Holdout Set\n", holdout_set)



	# Training
	for epoch in range(epochs):
		for sample in range(len(training_set)):
			for node in range(weights.shape[0]):
				print(np.sum(np.multiply(pre[sample,:], weights[node,:])))



	# Test to see the ouput is correct
	# print(np.append(pre, targets, axis=1))
	






if __name__ == '__main__':
    main()










