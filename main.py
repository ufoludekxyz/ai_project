import data
import network

trainData, testData = data.loadData()

# [input vector size, S1 neurons, S2 neurons, output]
net = network.Network([6, 2])

# (training_data, epochs, batch_size, eta, test_data)
#net.SGD(trainData, 100, 10, 0.9, test_data=None)
net.SGD(trainData, 1000, len(trainData), 0.9, test_data=testData)
