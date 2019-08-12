import numpy as np
import Mathematical_Functions as func


class Layer:

    def __init__(self, size, examples=1, desc=""):
        """Initializes Layer with size neurons of zeroes and appended constant 1 copied over number of
        examples

        Parameters:
        Size(int): Number of Neurons (including an extra constant 1 for bias)
        Examples (int,optional): Number of examples simultaneously stored in neuron, defaults to 1
        desc (str, optional): Short description, defaults to 1
        """
        self.examples = examples
        self.neurons = np.zeros((size + 1, examples))
        self.neurons[size, :] = np.ones_like(self.neurons[size, :])
        self.desc = desc

    def reset_layer(self, examples='-1'):
        """Resets Layer Back to zeroes and previous number of examples unless changed

        Parameters
        examples(int, optional): number of examples per neuron, defaults to previous examples
        """
        if examples != -1:
            self.examples = examples
            self.neurons = np.zeros((self.neurons.shape[0], examples))
            self.neurons[-1,:] = np.ones_like(self.neurons[-1,:])

    #def normal_dist(self, mean=0, stdev=1):
    #   """Sets neurons to normally distributed random values with mean and stdev

    #   Parameters:
    #   mean(float, optional): Mean of distribution, defaults to 0
    #   stdev(positive float, optional): Standard deviation of distribution, defaults to 1
    #   """
    #   self.neurons[:-1,:] = np.random.normal(loc=mean, scale=stdev, size=self.neurons[:-1,:].shape)
    #

    def upload_neuron_values(self, values):
        """Uploads new values to neurons in layer

        Parameters:
        values(numpy array) = new values of neurons, must be in correct shape
        """
        self.neurons[:-1,:] = values


class ConnectedLayerLink:

    def __init__(self, prev_layer, next_layer, activation=func.v_sigmoid, dactivation=func.v_dsigmoid, desc=""):
        """Initializes weights of connected layers as zeroes.

        Parameters:
        prev_layer and next_layer(Layer): layers in connected link.
        desc(str, optional): Description
        activation(func, optional): Activation Function used, defaults to sigmoid
        dactivation(func, optional): Derivative of Function used, defaults to dsigmoid
        """
        self.desc = desc
        self.weights = np.zeros((next_layer.neurons[0] - 1, prev_layer.neurons[0]))
        self.activate = activation
        self.dactivate = dactivation

    def normal_dist(self, mean=0, stdev=1):
        """Initializes weights to random numbers in a normal distribution with mean 0 and
         stdev 1

         Parameters:
         mean(float, optional): Mean of distribution, defaults to 0
         stdev(float, optional): Standard Deviation of distribution, defaults to 1
        """
        self.weights = np.random.normal(loc=mean, scale=stdev, size=self.weights)

    def propagate(self, prev_layer, next_layer):
        """Propagates values of prev_layer to next_layer and uploads values to next_layer

        Parameters:
        prev_layer(Layer): Previous Layer in Layer Link
        next_layer(Layer): Next Layer in Layer Link
        """
        next_layer.upload_neuron_values(
            self.activate(np.einsum(
                "np,pk -> nk", self.weights, prev_layer.neurons)))

    def dnext_dprev(self, prev_layer):
        """Returns 3 - dimensional numpy array matrix derivatives between next_layer
        and prev_layer in different examples

        Parameters:
        prev_layer(Layer): Previous Layer in LayerLink
        """
        return np.einsum("nk,np -> npk",
                         self.dactivate(
                             np.einsum("np,pk -> nk", self.weights, prev_layer.neurons)),
                         self.weights)

    def dnext_dweight(self, prev_layer):
        """Return 3 - dimensional numpy array matrix derivative between next_layer and
        the weights between prev_layer and next_layer in different examples

        Parameters:
        prev_layer(Layer): Previous layer in LayerLink
        """
        return np.einsum("nk,pk -> npk",
                         self.dactivate(
                             np.einsum("np,pk -> nk", self.weights, prev_layer.neurons)),
                         prev_layer.neurons)


class MLP:

    def __init__(self, input_size, output_size, desc=""):
        """Creates input and output layer for MLP

        Parameters:
        input_size(int): Number of neurons in the first layer
        output_size(int): Number of neurons in the ouptut layer
        desc(str, optional): Description
        """
        self.desc = desc
        self.input_layer = Layer(input_size)
        self.output_layer = Layer(output_size)
        self.layers = [self.input_layer, self.output_layer]
        self.links = []
        self.neuron_neuron_derivatives = []
        self.neuron_weight_derivatives = []
        self.output_weight_derivatives = []

    def reset_derivatives(self):
        self.neuron_neuron_derivatives = []
        self.neuron_weight_derivatives = []
        self.output_weight_derivatives = []

    def normal_dist(self, mean=0, stdev=1):
        """Assigns Weights to random number in a normal distribution

        Parameters:
        mean (float, optional): Mean of the distribution, defaults to 0
        stdev(float, optional): Standard Deviation of the distribution defaults to 1
        """
        for link in self.links:
            link.normal_dist(mean=mean, stdev=stdev)

    def add_hidden_layer(self, size, position):
        """Adds a layer in a selected position in the network

        Parameters:
        size(int): Number of neurons in added layer
        position(int): Position in MLP where layer is added
        """
        np.insert(self.layers, position, Layer(size))

    def gen_links(self):
        """Generates Links for Layers to connect"""
        for x in range(len(self.layers)-1):
            self.links.append(
                ConnectedLayerLink(self.layers[x], self.layers[x+1]))

    def calc_neurons(self, input_data):
        """Calculates Neurons and uploads them onto to network.

        Parameters:
        input_data(np-arrary): Input data in shape (neurons, examples)
        """
        example_number = input_data.shape[-1]
        for layer in self.layers:
            layer.reset_layer(examples=example_number)
        for x in range(len(self.links)):
            self.links[x].propogate(self.layers[x],self.layers[x+1])

    def compute_neuron_neuron_derivative(self):
        """Calculates derivatives between layers of all links and places them in
        a derivative_array attribute"""
        for x in range(len(self.links)):
            self.neuron_neuron_derivatives.append(self.links[x].dnext_dprev(self.layers[x]))

    def compute_neuron_weight_derivative(self):
        """Calculates derivatives between next_layer and weights for all layers links ad stores them
        in a neuron_weight_derivative_array attribute"""
        for x in range(len(self.links)):
            self.neuron_weight_derivatives.append(self.links[x].dnext_dweight(self.layers[x]))

    def compute_output_weight_derivative(self):
        factor = self.neuron_neuron_derivatives[-1]
        self.output_weight_derivatives.append(self.neuron_weight_derivatives[-1])
        for x in range(2, len(self.links)+ 1):
            self.output_weight_derivatives.append(
                np.einsum("npk,mnk -> mnpk",
                          self.neuron_weight_derivatives[-x], factor))
            factor = np.einsum("ijk,jhk -> ihk", factor, self.neuron_neuron_derivatives[-x])
        self.output_weight_derivatives.reverse()


class Trainer:

    def __init__(self, loss_func=func.quadratic_loss, dloss_func=func.v_dquadratic_loss):
        """Training algorithm used on network. Defaults to standard MSE loss

        Parameters:
        loss_func(func, optional): Loss Function used, defaults to quadratic_loss
        """
        self.loss = loss_func
        self.dloss = dloss_func

    def train_network_once(self, network, examples_input_data, learning_rate, examples_output_data):
        network.calc_neurons(examples_input_data)
        network.compute_neuron_neuron_derivative()
        network.compute_neuron_weight_derivative()
        network.compute_outpute_weight_derivative()

        cost = 0
        for i in range(examples_output_data.shape[-1]):
            cost += self.loss(examples_output_data[:, i], network.neurons[-1][:, i])
        cost = cost/examples_output_data.shape[-1]
        print(cost)

        a = self.dloss(examples_output_data, network.neurons[-1])
        for x in range(len(network.links)):
            network.links[x].weights -= learning_rate*np.einsum("ik,inpk -> np",
                                                                a,
                                                                network.output_weight_derivatives[x])/examples_input_data.shape[-1]

        cost = 0
        for i in range(examples_output_data.shape[-1]):
            cost += self.loss(examples_output_data[:, i], network.neurons[-1][:, i])
        cost = cost / examples_output_data.shape[-1]
        print(cost)