import data
import network

import numpy as np

trainData, testData = data.loadData()

# [input vector size, S1 neurons, S2 neurons, output]
net = network.Network([6, 2])

# (training_data, epochs, batch_size, eta, target, test_data)
#net.SGD(trainData, 100, 10, 0.9, test_data=None)
net.SGD(trainData, 10000, 10, 0.9, error_target=0.01,test_data=testData)

#for i in np.arange(1, 20, 1):
#    for j in np.arange(1, 20, 1):
#        net = network.Network([6, i, j, 2])
#        net.SGD(trainData, 1000, 10, 0.9, test_data=testData)
