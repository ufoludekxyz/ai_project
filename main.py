import data
import network

import numpy as np

trainData, testData = data.loadData()

# [input vector size, S1 neurons, S2 neurons, output]
net = network.Network([6, 2, 2])

# (training_data, epochs, batch_size, eta, test_data)
#net.SGD(trainData, 100, 10, 0.9, test_data=None)

"""
for j in range(10, len(trainData), 10):
    print("Batch size: ", j)
    for i in np.arange(0.01, 10, 0.01):
        print("Eta: ", i)
"""

net.SGD(trainData, 50, 10, 0.9, test_data=testData)
