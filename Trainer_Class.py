import numpy as np
import Mathematical_Functions as func


class Trainer:

    def __init__(self, loss_func=func.quadratic_loss, dloss_func=func.v_dquadratic_loss):
        """Training algorithm used on network. Defaults to standard MSE loss

        Parameters:
        loss_func(func, optional): Loss Function used, defaults to quadratic_loss
        """
        self.loss = loss_func
        self.dloss = dloss_func

    def train_network_once(self, network, examples_input_data, learning_rate, examples_output_data):
        """Trains the network once with given input and output data and a learning rate"""
        network.reset_derivatives()
        network.calc_neurons(examples_input_data)
        network.compute_neuron_neuron_derivative()
        network.compute_neuron_weight_derivative()
        network.compute_output_weight_derivative()

        a = self.dloss(examples_output_data, network.layers[-1].neurons)

        for x in range(len(network.links)):
            t = network.links[x].weights - learning_rate*np.einsum(
                "ik,inpk -> np",
                a,
                network.output_weight_derivatives[x])/examples_input_data.shape[-1]
            network.links[x].update_weights(t)
