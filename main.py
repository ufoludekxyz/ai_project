import numpy as np
from csv import reader
from tabulate import tabulate
import network

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

# Import data.tsv to dataList
dataList = dataImport('data.tsv')

# Create numpy array from dataList
dataList = np.array(dataList)

# Replace yes with 1
dataList = np.where(dataList == 'yes', 1, dataList)

# Replace no with 0
dataList = np.where(dataList == 'no', 0, dataList)

# Convert array of strings to array of floats
dataList = dataList.astype(float)

# Data normalization
normData = normalizeMinMax(dataList.T).T

# Splittint data into 2 groups, inputData (first 6 columns) and outputdata (last 2 columns)
inputData, outputData = normData[:,:6], normData[:,6:]

# Combining inputData and outputData in a single tuple
finalData = [(np.array(inputData[i], ndmin=2).T, np.array(outputData[i], ndmin=2).T) for i in range(0, len(outputData))]

#finalData = [((inputData[i]), (outputData[i])) for i in range(0, len(outputData))]

#class00 = np.array([i for i in finalData if i[1][0] == 0 and i[1][1] == 0])
#class01 = np.array([i for i in outputData if i[0] == 0 and i[1] == 1])
#class10 = np.array([i for i in outputData if i[0] == 1 and i[1] == 0])
#class11 = np.array([i for i in outputData if i[0] == 1 and i[1] == 1])

# [attributes, hidden neurons, output]
net = network.Network([6, 3, 2])

# (training_data, epochs, batch_size, eta, test_data)
net.SGD(finalData, 20, len(finalData), 0.9)
