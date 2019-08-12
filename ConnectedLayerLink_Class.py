import numpy as np
import Mathematical_Functions as func


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
        self.weights = np.zeros((next_layer.neurons.shape[0] - 1, prev_layer.neurons.shape[0]))
        self.activate = activation
        self.dactivate = dactivation

    def normal_dist(self, mean=0, stdev=1):
        """Initializes weights to random numbers in a normal distribution with mean 0 and
         stdev 1

         Parameters:
         mean(float, optional): Mean of distribution, defaults to 0
         stdev(float, optional): Standard Deviation of distribution, defaults to 1
        """
        self.weights = np.random.normal(loc=mean, scale=stdev, size=self.weights.shape)

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
                         self.weights)[:, :-1, :]

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

    def update_weights(self, weights):
        self.weights = weights
