import numpy as np
import MLP_Class as M

data = np.genfromtxt('train.csv', delimiter= ',', max_rows=1001)
data = data[1:, :]
data = np.swapaxes(data, 0, 1)

input_data = data[1:, :]
output_data = data[0, :].reshape(1,1000)

input_data_batch = input_data[:, :100]
output_data_batch = output_data[:, :100]

print(data)

test_network = M.MLP(input_data.shape[0], output_data.shape[0])
test_network.add_hidden_layer(20)
test_network.add_hidden_layer(20)
test_network.gen_links()
test_network.normal_dist()

test_trainer = M.Trainer()
for x in range(1,1000):
    test_trainer.train_network_once(test_network, input_data_batch, 0.001, output_data_batch)

