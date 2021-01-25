CPSC 481-04

Group: 
Michael Burdi
Jeffrey Saari
Justin Drouin


Project: Artificial Neural Network 


Intro/Algorithm:
The problem is we have a set of input vectors containing 10 features and we need to train a ANN to classify each of them into 1 of 10 known classes. 

For our ANN we used a multi-layer perceptron with one hidden layer containing 12 nodes. We also used a sigmoid function for our gradient descent and we used Mean Squared Error as a cost function. Our approach was to use all matrix multiplication to reduce the number of for loops. In Python, there is a fast array math library called NumPy that uses Python arrays as contiguous memory chunks. This makes working with NumPy arrays really fast compared to native Python arrays. We used this library to handle really fast multiplications of matrices. At first, we started with random weights and random biases for the I-to-H and H-to-O connections. Then as the network began to learn we updated the weights to new minima and started searching again. Eventually, we got the ANN to within 10% error. (Less than 10% was taking way too much time. Already we reached 5,341,843 epochs!!!




Setup/Installation:
To run the code requires Python3. It also requires Numpy Python library. 
A guide to install NumPy can be found here: https://numpy.org/install/

After the NumPy package is installed with all other Python binary files on your computer, in the terminal change to the project directory and execute the command: 

	python3 classifier.py



Note:
There is also a test.py function that was included which is where the weights were tested to verify the outputs.

An output.txt file has also been included that shows the latest output from running the program.

There are also two input files for each of the training sets provided from the professor. 




Features:
- Sigmoid/Sigmoid derivative functions provided
- Loop stops at either 10,000,000 epochs or if the user uses Cmd + C to interrupt the program it will exit and provide the final weights.





