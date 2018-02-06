"""
BSD 3-Clause License

Copyright (c) 2018, Jerrad Genson
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""

import numpy as np

from .trainer import Trainer
from learners.learner import LearnerType, InvalidLearnerTypeError


class Backpropagation(Trainer):
    """
    Backpropagation Trainer for the MultilayerPerceptron Learner.

    iterations: Number of iterations to run the training algorithm.
                If this is set too low, the algorithm might not converge
                on a solution. If set too high, it might take too long to
                run and/or overfit the data. Defaults to 1000.
    learning_rate: A float between 0 and 1 that determines the magnitude
                   of updates to the network's weights. A high learning
                   rate will cause the network to converge faster, but
                   might negatively impact the precision/solution quality.
                   Defaults to 0.25.
    randomize: A flag that indicates whether or not to randomize inputs
               and targets. This can improve the speed at which the
               training algorithm converges. Default value is False.
    momentum: The amount of "momentum" to conserve during training as a float
              between 0 and 1. Defaults to 0.9.

    """

    def __init__(self, iterations=1000, learning_rate=0.25, randomize=False,
                 momentum=0.9):

        self.learning_rate = learning_rate
        self.randomize = randomize
        self.momentum = momentum
        return super().__init__(iterations)

    @staticmethod
    def _error_gradient(beta, activations, previous_layer, previous_delta):
        """
        Compute the error gradient for a hidden layer.

        Args
          beta: Learning rate of the network.
          activations: Output values for the current layer.
          previous_layer: Weights of the previous layer.
          previous_delta: Delta value for the previous layer's error gradient.

        Returns
          A delta value for the current layer's error gradient.

        """

        return (activations * beta * (1.0 - activations) *
                (np.dot(previous_delta, np.transpose(previous_layer))))

    def _backpropagate(self, learner, activations, layers, layers_updates,
                       previous_delta):

        """
        Recursively perform backpropagation on each network layer.

        Args
          learner: An instance of learners.learner.Learner.
          activations: A sequence of all the network's activations.
          layers: A sequence of the network's layer weights.
          layers_updates: A sequence of the layer weight updates computed by
                          previous runs of backpropagation.
          previous_delta: The delta values computed for the previous layer.

        Returns
          A tuple of (layer_weights, updates) where layers_weights is a list of
          new network layer weights and updates is a list of new layer weight
          updates.

        """

        layer_activations = activations[-1]
        layer_inputs = activations[-2]
        layer_weights = layers[-2]
        previous_weights = layers[-1]
        layer_delta = self._error_gradient(learner.beta,
                                           layer_activations,
                                           previous_weights,
                                           previous_delta)

        if len(layers) == 2 and learner.bias:
            layer_delta = layer_delta[:, :-1]

        updates = (self.learning_rate *
                   np.dot(np.transpose(layer_inputs), layer_delta) +
                   self.momentum * layers_updates[-1])

        layer_weights -= updates
        if len(layers) == 2:
            # Basis step
            return [layer_weights], [updates]

        else:
            # Inductive step
            next_layer, next_updates = self._backpropagate(learner,
                                                           activations[:-1],
                                                           layers[:-1],
                                                           layers_updates[:-1],
                                                           layer_delta)

            return next_layer + [layer_weights], next_updates + [updates]

    def _train(self, learner, inputs, targets):
        """
        Train the neural network using backpropagation.
        Training happens en batch, which means all the training data is fed to
        the algorithm at once. Mutates self.hidden_weights and
        self.output_weights.

        Args
          learner: An instance of learners.learner.Learner.
          inputs: Training inputs to the network as a numpy array of arrays,
                  where each inner array is one set of inputs.
          targets: Target outputs for the network as a numpy array of arrays,
                   where each inner array is one set of target outputs. Target
                   arrays must match the order of input arrays.

        Returns
          Sum of squares error of the last network recall on the input data.

        """

        # Add the inputs that match the bias node
        training_dataset_rows = np.shape(inputs)[0]
        inputs = learner._concat_bias(inputs)

        # Compute the initial order of input and target nodes so we can
        # randomize them if we so choose.
        node_order = list(range(training_dataset_rows))

        # Create arrays to store update values for weight vectors.
        layers_updates = [np.zeros(layer.shape) for layer in learner.layers]
        for iteration in range(self.iterations):
            outputs = learner._recall(inputs, False)
            error = 0.5 * np.sum((outputs - targets)**2)

            # Compute the output layer error gradient for different activation functions.
            if learner.learner_type == LearnerType.REGRESSION:
                deltao = (outputs - targets) / training_dataset_rows

            elif learner.learner_type == LearnerType.CLASSIFICATION:
                deltao = learner.beta * (outputs - targets) * \
                    outputs * (1.0 - outputs)

            elif learner.learner_type == LearnerType.ONE_OF_N:
                deltao = ((outputs - targets) *
                          (outputs * -outputs + outputs) /
                          training_dataset_rows)

            else:
                raise InvalidLearnerTypeError(
                    'learner_type not member of LearnerType')

            output_updates = (self.learning_rate *
                              np.dot(np.transpose(learner.activations[-2]), deltao) +
                              self.momentum * layers_updates[-1])

            new_output_layer = learner.layers[-1] - output_updates
            new_layers, new_updates = self._backpropagate(learner,
                                                          learner.activations[:-1],
                                                          learner.layers,
                                                          layers_updates[:-1],
                                                          deltao)

            learner.layers = new_layers + [new_output_layer]
            layers_updates = new_updates + [output_updates]
            if self.randomize:
                # Randomize order of input vector and update target vector correspondingly.
                np.random.shuffle(node_order)
                inputs = inputs[node_order, :]
                targets = targets[node_order, :]

        return error
