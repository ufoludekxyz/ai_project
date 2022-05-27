import data
import network

import numpy as np

trainData, testData = data.loadData()

# [input vector size, S1 neurons, S2 neurons, output]
#net = network.Network([6, 2, 2])

# (training_data, epochs, batch_size, eta, test_data)
#net.SGD(trainData, 100, 10, 0.9, test_data=None)

for i in np.arange(10, len(trainData), 10):
    for j in np.arange(0.01, 15, 0.01):
        net = network.Network([6, 2])
        net.SGD(trainData, 200, i, j, test_data=testData)
