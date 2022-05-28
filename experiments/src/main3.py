import data
import network3 as network

import numpy as np
import pandas as pd

trainData, testData = data.loadData()

# [input vector size, S1 neurons, S2 neurons, output]
#net = network.Network([6, 2, 2, 2])

# (training_data, epochs, batch_size, eta, target, test_data)
#net.SGD(trainData, 100, 10, 0.9, test_data=None)
#net.SGD(trainData, 10000, 10, 0.9, error_target=0.10,test_data=testData)

results = []

net = network.Network([6, 2])
results.append(net.SGD(trainData, 100000, 10, 0.9, error_target=0.07, test_data=testData))

results = pd.DataFrame(results)
results.to_csv('experiments/results3.csv', index=None, header=None)
