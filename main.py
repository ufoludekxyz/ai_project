import data
import network

trainData, testData = data.loadData()

# [attributes, hidden neurons, output]
net = network.Network([6, 3, 2])

# (training_data, epochs, batch_size, eta, test_data)
net.SGD(trainData, 2000, 10, 0.9, test_data=None)
#net.SGD(trainData, 200, 10, 0.9, test_data=testData)
