import data
import network1 as network

import numpy as np
import pandas as pd


trainData, testData = data.loadData()

# [input vector size, S1 neurons, S2 neurons, output]
net = network.Network([6,2])

# (training_data, epochs, batch_size, eta, target, test_data)
#net.SGD(trainData, 100, 10, 0.9, test_data=None)
#net.SGD(trainData, 100000, 1, 0.1, error_target=0.179,test_data=testData)
results = []

for i in np.arange(1, len(trainData)+1, 1):
    net = network.Network([6, 2])
    results.append(net.SGD(trainData, 10000, i, 0.3, error_target=0.179, test_data=testData))

results = pd.DataFrame(results)
results.to_csv('../batch_size.csv', index=None, header=None)
