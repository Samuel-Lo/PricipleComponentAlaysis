import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv


# This function read the CSV file first and call functions to obtain the value we want
def start():
    inputData = pd.read_csv('MNIST_100.csv') # read input
    Y = inputData['class'].values # get class label
    inputData = inputData.drop('class', axis=1)
    X = inputData

    stdX = calculateMean(X, X.mean(axis = 0))

    eVals, eVecs = calculateEigen(stdX)

    EV = []
    projectM = calculateProjectMatrix(EV, eVals, eVecs)

    plotPCA(stdX, projectM, Y)

# This function compute the mean of input data
def calculateMean(X, meanX):
    meanX = X.mean(axis=0)
    stdX = X - meanX
    return stdX

# This function compute the eigenvector and eigencalue
def calculateEigen(stdX):
    covX = np.cov(stdX.T)
    return np.linalg.eigh(covX)

# This function compute the projected matrix
def calculateProjectMatrix(EV, eVals, eVecs):
    for i in range(0, len(eVals)):
        EV.append((np.abs(eVals[i]), eVecs[:, i]))

    EV.sort(key=lambda x: x[0], reverse=True)
    print(len(EV))
    EP1 = np.reshape(EV[0][1], (-1,1))
    EP2 = np.reshape(EV[1][1], (-1,1))
    projectM = np.hstack((EP1,EP2))
    return projectM

# This function plot the result in 2D
def plotPCA(stdX, projectM, Y):
    with open('MSByteColorPCAData100.csv', 'w+', newline='') as myfile:
        wr = csv.writer(myfile,delimiter=',', quoting=csv.QUOTE_NONE, quotechar='', escapechar = '\\')
        markerColors = ['black', 'orange', 'green', 'blue', 'pink', 'lightgreen', 'gray', 'purple', 'brown', 'red']
        markers = ['$1$', '$2$', '$3$', '$4$', '$5$', '$6$', '$7$', '$8$', '$9$']
        pcaX = np.dot(stdX, projectM)
        labels = np.unique(Y)
        plotAtt = zip(labels, markers, markerColors)
        for label, marker, color in plotAtt:
            l = (Y == label)
            x = pcaX[l, 0]
            y = pcaX[l, 1]
            data = []
            print(x)
            for index in range(0, len(x)):
                data.append(label)
                data.append(x[index])
                data.append(y[index])
                wr.writerow(data)
                data.clear()
            plt.scatter(x, y, c=color, marker=marker, edgecolors='none')
        plt.xlabel('Principle Component 1')
        plt.ylabel('Principle Component 2')
        plt.show()


if __name__ == "__main__":
    start()
