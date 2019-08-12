import numpy as np
import Trainer_Class as T
import MLP_Class as M


def compare(x, y):
    if x == y:
        return 1
    if x != y:
        return 0


v_compare = np.vectorize(compare)


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

test_trainer = T.Trainer()
for x in range(1, 10):
    for y in range(1,100):
        test_trainer.train_network_once(test_network, input_data_batch, 0.001, output_data_batch)
    test_network.calc_neurons(input_data_batch)
    predictions = np.around(test_network.return_output_neurons())
    print(np.sum(v_compare(predictions, output_data_batch)))
