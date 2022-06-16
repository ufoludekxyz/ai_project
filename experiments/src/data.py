from sklearn.model_selection import train_test_split
from csv import reader
import numpy as np

# Import function
def dataImport(name):
    with open(name, 'r', encoding='utf-16') as file:
        return [line for line in reader(file, delimiter='\t')]

# Normalization function
def normalizeMinMax(table):
    for row in range(0, len(table)):
        min_val, max_val = min(table[row]), max(table[row])
        table[row] = [(1 - 0) * (col - min_val) / (max_val - min_val) for col in table[row]]
    return table

# Data loader function
def loadData():
    # Import acute.tsv to dataFile
    dataFile = dataImport('../../acute.tsv')

    # Create numpy array from dataList
    dataFile = np.array(dataFile)

    # Convert array of strings to array of floats
    dataFile = dataFile.astype(float)

    # Data normalization
    dataFile = normalizeMinMax(dataFile.T).T

    # Data splitting into training and test data
    trainData, testData = train_test_split(dataFile, test_size=0.2, random_state=25)

    # Splitting data into 2 groups, inputData (first 6 columns) and outputdata (last 2 columns)
    testIn, testOut = testData[:,:6], testData[:,6:]
    trainIn, trainOut = trainData[:,:6], trainData[:,6:]

    # Combining inputData and outputData in a single tuple
    trainData = [(np.array(trainIn[i], ndmin=2).T, np.array(trainOut[i], ndmin=2).T) for i in range(0, len(trainOut))]
    testData = [(np.array(testIn[i], ndmin=2).T,np.array(testOut[i], ndmin=2).T) for i in range(0, len(testOut))]

    return (trainData, testData)
