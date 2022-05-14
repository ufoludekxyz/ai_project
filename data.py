import numpy as np
from csv import reader
from tabulate import tabulate

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

print(tabulate(normData))
