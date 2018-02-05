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

Inspired by code in "Machine Learning: An Algorithmic Perspective" by
Dr. Stephen Marsland.

"""

import logging

import numpy as np

from .learner import LearnerType, Learner, InvalidLearnerTypeError

logger = logging.getLogger(__name__)


class MultilayerPerceptron(Learner):
    """
    A multilayer perceptron learner.
    Multilayer perceptron is a type of neural network that features an input
    layer, one to two hidden layers, and an output layer, which are trained
    by the backpropagation algorithm.

    Args
      input_node_count: Number of inputs to the network.
      hidden_node_count: Number of hidden nodes in the network's hidden layer.
      classes: A sequence of classification names. This must be defined for
               ONE_OF_N learners. Defaults to None.
      beta: A constant in the logistic activation equation. Defaults to 1.
      learner_type: Activation function to use at the output nodes. Must be
                   a member of LearnerType. Defaults to CLASSIFICATION.
      bias: Add bias nodes of -1 into the inputs array. Defaults to True.
      second_hidden_layer_node_count: Number of hidden nodes in the network's
                                      second hidden layer. Any value less than
                                      or equal to 0 will construct a network
                                      without a second hidden layer.
                                      Defaults to 0.

    """

    def __init__(self, shape, classes=None, beta=1,
                 learner_type=LearnerType.CLASSIFICATION, bias=True):

        self.beta = beta
        self.learner_type = learner_type
        self.bias = bias
        self.shape = shape
        self.activations = []
        if learner_type not in (LearnerType.CLASSIFICATION, LearnerType.REGRESSION, LearnerType.ONE_OF_N):
            raise InvalidLearnerTypeError('learner_type not member of LearnerType')

        # Initialise network
        self.layers = []
        for layer_index, node_count in enumerate(shape[1:]):
            previous_count = shape[layer_index]
            if 0 < layer_index < len(shape) - 2 and bias:
                node_count += 1

            weights = ((np.random.rand(previous_count + 1, node_count) - 0.5) *
                       2 / np.sqrt(previous_count))

            self.layers.append(weights)

        return super().__init__(classes=classes, learner_type=learner_type)

    def train_with_early_stopping(self, training_inputs, training_targets,
                                  validation_inputs, validation_targets,
                                  learning_rate, iterations=100, max_epoch=-1,
                                  momentum=0.9):
        """
        Train the neural network using backpropagation and early stopping.
        Stop training when the validation set error consistently increases.

        Args
          training_inputs: Training inputs to the network as a numpy array of
                           arrays, where each inner array is one set of inputs.
          training_targets: Target outputs for the network as a numpy array of
                            arrays, where each inner array is one set of target
                            outputs. Target arrays must match the order of input
                            arrays.
          validation_inputs: Similar to training_inputs, but used to determine
                             when the early stopping condition has been met.
          validation_targets: Similar to trainings_targets, but used to determine
                              when the early stopping condition has been met.
          learning_rate: A float between 0 and 1 that determines the magnitude
                         of updates to the network's weights. A high learning
                         rate will cause the network to converge faster, but
                         might negatively impact the precision/solution quality.
          iterations: Number of iterations to run the training algorithm per epoch.
                      Defaults to 100.
          max_epoch: Maximum number of "runs" of the training algorithm. A value
                     <= 0 indicates no limit. Defaults to -1.
          momentum: The amount of "momentum" to conserve during training as a float
                    between 0 and 1. Defaults to 0.9.

        Returns
          Sum of squares error of the last network recall on the validation data.

        """

        valid = np.concatenate(
            (validation_inputs, -np.ones((np.shape(validation_inputs)[0], 1))), axis=1)

        oldest_error = 0
        previous_error = 0
        current_error = 0
        current_epoch = 0
        while True:
            # Iterate at least three times in order to properly initialize error
            # variables before early stopping.
            if previous_error - current_error < 0.001 and oldest_error - previous_error < 0.001 and current_epoch > 2:
                break

            if current_epoch > max_epoch and max_epoch > 0:
                break

            current_epoch += 1
            logger.info(current_epoch)
            self._train(training_inputs, training_targets,
                        learning_rate, iterations, momentum)
            oldest_error = previous_error
            previous_error = current_error
            output_value = self.__recall(False, valid)
            current_error = (0.5 * np.sum((validation_targets - output_value)**2))

        logger.info("Stopped", current_error, previous_error, oldest_error)
        return current_error

    def _error_gradient(self, activations, previous_layer, previous_delta):
        """
        Compute the error gradient for a hidden layer.

        Args
          activations: Output values for the current layer.
          previous_layer: Weights of the previous layer.
          previous_delta: Delta value for the previous layer's error gradient.

        Returns
          A delta value for the current layer's error gradient.

        """

        return (activations * self.beta * (1.0 - activations) *
                (np.dot(previous_delta, np.transpose(previous_layer))))

    def _backpropagate(self, learning_rate, momentum, activations,
                       layers, layers_updates, previous_delta):

        layer_activations = activations[-1]
        layer_inputs = activations[-2]
        layer_weights = layers[-2]
        previous_weights = layers[-1]
        layer_delta = self._error_gradient(layer_activations,
                                           previous_weights,
                                           previous_delta)

        if len(layers) == 2 and self.bias:
            layer_delta = layer_delta[:, :-1]

        updates = (learning_rate *
                   np.dot(np.transpose(layer_inputs), layer_delta) +
                   momentum * layers_updates[-1])

        layer_weights -= updates
        if len(layers) == 2:
            # Basis step
            return [layer_weights], [updates]

        else:
            # Inductive step
            next_layer, next_updates = self._backpropagate(learning_rate,
                                                           momentum,
                                                           activations[:-1],
                                                           layers[:-1],
                                                           layers_updates[:-1],
                                                           layer_delta)

            return next_layer + [layer_weights], next_updates + [updates]

    def _train(self, inputs, targets, learning_rate, iterations, randomize=False,
               momentum=0.9):
        """
        Train the neural network using backpropagation.
        Training happens en batch, which means all the training data is fed to
        the algorithm at once. Mutates self.hidden_weights and
        self.output_weights.

        Args
          inputs: Training inputs to the network as a numpy array of arrays,
                  where each inner array is one set of inputs.
          targets: Target outputs for the network as a numpy array of arrays,
                   where each inner array is one set of target outputs. Target
                   arrays must match the order of input arrays.
          learning_rate: A float between 0 and 1 that determines the magnitude
                         of updates to the network's weights. A high learning
                         rate will cause the network to converge faster, but
                         might negatively impact the precision/solution quality.
          iterations: Number of iterations to run the training algorithm.
                      If this is set too low, the algorithm might not converge
                      on a solution. If set too high, it might take too long to
                      run and/or overfit the data.
          randomize: A flag that indicates whether or not to randomize inputs
                     and targets. This can improve the speed at which the
                     training algorithm converges. Default value is False.
          momentum: The amount of "momentum" to conserve during training as a float
                    between 0 and 1. Defaults to 0.9.

        Returns
          Sum of squares error of the last network recall on the input data.

        """

        # Add the inputs that match the bias node
        training_dataset_rows = np.shape(inputs)[0]
        inputs = self._concat_bias(inputs, training_dataset_rows)

        # Compute the initial order of input and target nodes so we can
        # randomize them if we so choose.
        node_order = list(range(training_dataset_rows))

        # Create arrays to store update values for weight vectors.
        layers_updates = [np.zeros(layer.shape) for layer in self.layers]
        for iteration in range(iterations):
            outputs = self.__recall(False, inputs)
            error = 0.5 * np.sum((outputs - targets)**2)

            # Compute the output layer error gradient for different activation functions.
            if self.learner_type == LearnerType.REGRESSION:
                deltao = (outputs - targets) / training_dataset_rows

            elif self.learner_type == LearnerType.CLASSIFICATION:
                deltao = self.beta * (outputs - targets) * \
                    outputs * (1.0 - outputs)

            elif self.learner_type == LearnerType.ONE_OF_N:
                deltao = ((outputs - targets) *
                          (outputs * -outputs + outputs) /
                          training_dataset_rows)

            else:
                raise InvalidLearnerTypeError(
                    'learner_type not member of LearnerType')

            output_updates = (learning_rate *
                              np.dot(np.transpose(self.activations[-2]), deltao) +
                              momentum * layers_updates[-1])

            new_output_layer = self.layers[-1] - output_updates
            new_layers, new_updates = self._backpropagate(learning_rate,
                                                          momentum,
                                                          self.activations[:-1],
                                                          self.layers,
                                                          layers_updates[:-1],
                                                          deltao)

            self.layers = new_layers + [new_output_layer]
            layers_updates = new_updates + [output_updates]
            if randomize:
                # Randomize order of input vector and update target vector correspondingly.
                np.random.shuffle(node_order)
                inputs = inputs[node_order, :]
                targets = targets[node_order, :]

        return error

    def _logistic(self, layer_values):
        """
        Logistic activation function.

        Args
          layer_values: A numpy array of network layer input or output values.

        Returns
          The value of the weights with logistic activation applied.

        """

        return 1.0 / (1.0 + np.exp(-self.beta * layer_values))

    def _concat_bias(self, layer_values, rows):
        """
        Concatenate bias node to network layer values.

        Args
          layer_values: A numpy array of network layer input or output values.
          rows: The number of rows in layer_values. If there is only one set of
                values in the array, this will be 1.

        Returns
          layer_values with a bias node contatenated onto the end.

        """

        return np.concatenate((layer_values, -np.ones((rows, 1))), axis=1)

    def __recall(self, bias, inputs):
        """
        Perform a recall on a given set of inputs.
        In other words, run the network to make a prediction or classification.

        Args
          inputs: Input data to the network as a numpy array of arrays, where
                  each inner array is one set of inputs.

        Returns
          A numpy array of arrays representing the network's outputs, where each
          inner array corresponds to an inner array in the inputs.

        """

        dataset_rows = np.shape(inputs)[0]
        if bias:
            # Add the inputs that match the bias node.
            inputs = self._concat_bias(inputs, dataset_rows)

        self.activations = [inputs]

        # Compute activations for each network layer.
        for layer_index, layer in enumerate(self.layers):
            layer_activations = np.dot(self.activations[-1], layer)
            if layer_index != len(self.layers) - 1:
                # Do not apply logistic activation to output layer.
                layer_activations = self._logistic(layer_activations)

            if layer_index == 0 and self.bias:
                # Concatendate bias node to first hidden layer.
                layer_activations = self._concat_bias(layer_activations, dataset_rows)

            self.activations.append(layer_activations)

        if self.learner_type == LearnerType.REGRESSION:
            # Use linear activation (null activation function) for regression.
            pass

        elif self.learner_type == LearnerType.CLASSIFICATION:
            # Use logistic activation function for classification.
            self.activations[-1] = self._logistic(self.activations[-1])

        elif self.learner_type == LearnerType.ONE_OF_N:
            # Use soft-max activation function for 1-of-N classification.
            normalisers = np.sum(np.exp(self.activations[-1]), axis=1) * \
                np.ones((1, np.shape(self.activations[-1])[0]))

            self.activations[-1] = np.transpose(np.transpose(np.exp(self.activations[-1])) / normalisers)

        else:
            raise InvalidLearnerTypeError(
                'learner_type not member of LearnerType')

        return self.activations[-1]

    def _recall(self, inputs):
        return self.__recall(self.bias, inputs)

    _recall.__doc__ = __recall.__doc__

    def generate_confusion_matrix(self, inputs, targets):
        """
        Generate a confusion matrix to show how well the network classifies data.

        Args
          inputs: Training inputs to the network as a numpy array of arrays,
                  where each inner array is one set of inputs.
          targets: Target outputs for the network as a numpy array of arrays,
                   where each inner array is one set of target outputs. Target
                   arrays must match the order of input arrays.

        Returns
          A numpy array of arrays representing the confusion matrix data.

        """

        # Run the network forward to get the outputs.
        outputs = self._recall(inputs)

        # Compute the number of distinct classifications.
        classifications = np.shape(targets)[1]
        if classifications == 1:
            # Logistic classification
            classifications = 2
            outputs = np.where(outputs > 0.5, 1, 0)

        else:
            # 1-of-N encoding
            outputs = np.argmax(outputs, 1)
            targets = np.argmax(targets, 1)

        # Initialize the confusion matrix arrays.
        confusion_matrix = np.zeros((classifications, classifications))

        # Read classification data into the confusion matrix.
        for i in range(classifications):
            for j in range(classifications):
                confusion_matrix[i, j] = np.sum(np.where(outputs == i, 1, 0)
                                                * np.where(targets == j, 1, 0))

        logger.info("Confusion matrix is:")
        logger.info(str(confusion_matrix))
        logger.info("Percentage Correct: ", np.trace(
            confusion_matrix) / np.sum(confusion_matrix) * 100)

        return confusion_matrix
