import numpy as np


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
        self.neurons[-1, :] = np.ones_like(self.neurons[-1, :])
        self.desc = desc

    def reset_layer(self, examples='-1'):
        """Resets Layer Back to zeroes and previous number of examples unless changed

        Parameters
        examples(int, optional): number of examples per neuron, defaults to previous examples
        """
        if examples != -1:
            self.examples = examples
            self.neurons = np.zeros((self.neurons.shape[0], examples))
            self.neurons[-1, :] = np.ones_like(self.neurons[-1, :])
        else:
            self.neurons = np.zeros((self.neurons.shape[0], 1))
            self.neurons[-1, :] = np.ones_like(self.neurons[-1, :])

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
        self.neurons[:-1, :] = values
