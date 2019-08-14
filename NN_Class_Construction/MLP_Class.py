import numpy as np
import Layer_Class as L
import ConnectedLayerLink_Class as C


class MLP:

    def __init__(self, input_size, output_size, desc=""):
        """Creates input and output layer for MLP

        Parameters:
        input_size(int): Number of neurons in the first layer
        output_size(int): Number of neurons in the ouptut layer
        desc(str, optional): Description
        """
        self.desc = desc
        self.input_layer = L.Layer(input_size)
        self.output_layer = L.Layer(output_size)
        self.layers = [self.input_layer, self.output_layer]
        self.links = []
        self.neuron_neuron_derivatives = []
        self.neuron_weight_derivatives = []
        self.output_weight_derivatives = []

    def reset_derivatives(self):
        """Empties derivative arrays before being recalculated"""
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

    def add_hidden_layer(self, size, position=1):
        """Adds a layer in a selected position in the network

        Parameters:
        size(int): Number of neurons in added layer
        position(int): Position in MLP where layer is added
        """
        self.layers = np.insert(self.layers, position, L.Layer(size))

    def gen_links(self):
        """Generates Links for Layers to connect"""
        for x in range(len(self.layers)-1):
            self.links.append(
                C.ConnectedLayerLink(self.layers[x], self.layers[x+1]))

    def calc_neurons(self, input_data):
        """Calculates Neurons and uploads them onto to network.

        Parameters:
        input_data(np-arrary): Input data in shape (neurons, examples)
        """
        example_number = input_data.shape[-1]
        for layer in self.layers:
            layer.reset_layer(examples=example_number)
        #print(input_data)
        #self.layers[0].neurons[:-1, :] = input_data
        self.layers[0].upload_neuron_values(input_data)
        #print(self.layers[0].neurons[:10, :10])
        for x in range(len(self.links)):
            self.links[x].propagate(self.layers[x], self.layers[x+1])

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
        """Computes output to weight derivatives and places them in a list of 4 dimensional arrays"""
        factor = self.neuron_neuron_derivatives[-1]
        a = self.neuron_weight_derivatives[-1].shape[0]
        self.output_weight_derivatives.append(np.einsum("npk, nh -> hnpk",
                                                        self.neuron_weight_derivatives[-1], np.identity(a)))
        for x in range(2, len(self.links)+ 1):
            self.output_weight_derivatives.append(
                np.einsum("npk,mnk -> mnpk",
                          self.neuron_weight_derivatives[-x], factor))
            factor = np.einsum("ijk,jhk -> ihk", factor, self.neuron_neuron_derivatives[-x])
        self.output_weight_derivatives.reverse()

    def return_output_neurons(self):
        """Returns output neurons"""
        return self.layers[-1].neurons[:-1, :]
