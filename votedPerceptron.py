'''
Implement the batch gradient descent alg and tune r to find convergence
Convergence can be found by examinging the norm of the weight vector difference ||w_t-w_(t-1)|| at each step t.
    If ||w_t-w_(t-1)|| < 1e-6, then converges.
Weight vector is initialized to 0
Start with high r, and decrease by half each time until you find convergence
Report weight vector and learning rate.  Record cost function value of the training data at each step
Draw figure showing how cost function changes with time steps.  Use final weight vector to calc the cost function value of the test data

'''

from Dataset import *
from Utility import *
import numpy as np
import random as rand


# This runs batch gradient descent on our example, called by the main function
def trainVotedPerceptron(max_epoch, r):

    # Grab the training data (code recycled from the Decision Tree Project
    trainingData, testingData = getPerceptron_Data();

    prevIterWeights, curIterWeights = initializeWeights(trainingData)


    weightList = []

    for epoch in range (1, max_epoch):
        randomList = rand.sample(trainingData.getExampleList(), len(trainingData.getExampleList()))

        Cm = 1
        for example in randomList:
            yi, xi = getYiXi(example)
            if yi == 0.0:
                yi = -1.0
            if yi* np.dot(np.transpose(curIterWeights), np.array(xi)) <= 0:
                weightList.append((curIterWeights, Cm))
                curIterWeights = curIterWeights + r * yi * np.array(xi)
                Cm = 1
            else:
                Cm = Cm+1

    weightList.append((curIterWeights,Cm))
    return weightList

def testVotedPerceptron(weights):
    # Grab the training data (code recycled from the Decision Tree Project
    trainingData, testingData = getPerceptron_Data();


    totalTests = 0.0
    totalCorrect = 0.0

    for example in testingData.getExampleList():
        yi, xi = getYiXi(example)
        if yi == 0.0:
            yi = -1.0
        sumTotal = 0
        for weight in weights:
            ci = weight[1]
            sumTotal += ci * np.sign(np.dot(np.transpose(weight[0]), np.array(xi)))
        prediction = np.sign(sumTotal)
        if prediction == np.sign(yi):
            totalCorrect += 1.0
        totalTests += 1.0

    accuracy = totalCorrect / totalTests

    return accuracy

if __name__ == "__main__":
    max_epoch = 10
    r = 0.1
    votedPerceptronWeights = trainVotedPerceptron(max_epoch, r)
    votedPerceptronAccuracy = testVotedPerceptron(votedPerceptronWeights)
    print "Voted Perceptron Weight: " + str(votedPerceptronWeights)
    print "Accuracy in testing: " + str(votedPerceptronAccuracy)