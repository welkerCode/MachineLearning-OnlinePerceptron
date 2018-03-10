import numpy as np
import matplotlib.pyplot as plt
from Dataset import *
from Example import *



# Returns a boolean that tells whether the data has converged or not
def converges(newWeight, oldWeight, epsilon):
    diff = newWeight-oldWeight
    sum = 0
    for element in diff:
        sum += element**2
    error = np.sqrt(sum)
    if error < epsilon:
        return True
    else:
        return False

def getGD_Data():
    # Grab the training data (code recycled from the Decision Tree Project)
    trainingData = Dataset('dataset-hw2/regression/train.csv')
    testingData = Dataset('dataset-hw2/regression/test.csv')
    return trainingData, testingData

def getPerceptron_Data():
    # Grab the training data (code recycled from the Decision Tree Project)
    trainingData = Dataset('dataset-hw2/classification/train.csv')
    testingData = Dataset('dataset-hw2/classification/test.csv')
    return trainingData, testingData

def initializeWeights(data):
    featureVector = []
    for attribute in data.getAttributeList():
        featureVector.append(0)
    featureVector.append(0) # For the bias term
    return np.array(featureVector), np.array(featureVector)

# Quickly extracts yi and xi when given an example from the training or testing data
def getYiXi(example):
    # Get yi from the label
    yi = float(example.getLabel())

    # Get all of the features, place in xi
    xi_string = example.getAttributes()
    xi = []
    for feature in xi_string:
        xi.append(float(feature))
    xi.append(1.0) # For the bias term
    return yi, xi

# This function evaluates the cost function J(w) for a given timestep
def calcCost(data, weights):

    # Initialize something to hold the sum of the following calculations
    sumTotal = 0.0

    # For every example in the training data
    for example in data.getExampleList():

        yi, xi = getYiXi(example)

        # sum((yi - w^T xi)^2)
        sumTotal += (yi - np.dot(np.transpose(weights), np.array(xi)))**2

    # Have to take the negative of the sum, and place in our new gradient vector
    Jw = .5*sumTotal

    return Jw

def plotCost(costs, title):

    # Example taken from matplotlib.org/examples/pylab_examples/simple_plot.html
    t = np.arange(0, len(costs), 1)

    plt.plot(t,costs)
    plt.xlabel('time step')
    plt.ylabel('cost')
    plt.title(title)
    plt.grid(True)
    plt.savefig(title+".png")
    plt.show