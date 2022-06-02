import data
import network6 as network

import numpy as np
import pandas as pd


trainData, testData = data.loadData()

# [input vector size, S1 neurons, S2 neurons, output]
#net = network.Network([6,2])

# (training_data, epochs, batch_size, eta, target, test_data)
#net.SGD(trainData, 100, 10, 0.9, test_data=None)
#net.SGD(trainData, 100000, 1, 0.1, error_target=0.179,test_data=testData)
results = []

for i in range(2, 21, 1):
    net = network.Network([6, 1, i, 2])
    results.append(net.SGD(trainData, 10000, 1, 0.1, error_target=0.18, test_data=testData))

results = pd.DataFrame(results)
results.to_csv('../epo_S1=1_S2_err_acc.csv', index=None, header=None)
