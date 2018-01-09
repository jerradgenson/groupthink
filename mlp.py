# Code from Chapter 4 of Machine Learning: An Algorithmic Perspective (2nd Edition)
# by Stephen Marsland (http://stephenmonika.net)

# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes, but please maintain the name of the original author.
# This code comes with no warranty of any kind.

# Stephen Marsland, 2008, 2014

import logging
from enum import Enum

import numpy as np


OutputType = Enum('OutputType', 'LINEAR LOGISTIC SOFTMAX', module=__name__)
logger = logging.getLogger(__name__)


class MultilayerPerceptron:
    """
    A multilayer perceptron learner.
    Multilayer perceptron is a type of neural network that features an input
    layer, one to two hidden layers, and an output layer, which are trained
    by the backpropagation algorithm.

    Args
      input_node_count: Number of inputs to the network.
      hidden_node_count: Number of hidden nodes in the network's hidden layer.
      output_node_count: Number of output nodes.
      beta: A constant in the logistic activation equation. Defaults to 1.
      momentum: The amount of "momentum" to conserve during training as a float
                between 0 and 1. Defaults to 0.9.
      output_type: Activation function to use at the output nodes. Must be
                   a member of OutputType. Defaults to LOGISTIC.

    """

    def __init__(self, input_node_count, hidden_node_count, output_node_count,
                 beta=1, momentum=0.9, output_type=OutputType.LOGISTIC):

        self.beta = beta
        self.momentum = momentum
        self.output_type = output_type

        # Initialise network
        self.hidden_weights = (np.random.rand(
            input_node_count + 1, hidden_node_count) - 0.5) * 2 / np.sqrt(input_node_count)

        self.output_weights = (np.random.rand(hidden_node_count + 1,
                                              output_node_count) - 0.5) * 2 / np.sqrt(hidden_node_count)

    def train_with_early_stopping(self, training_inputs, training_targets,
                                  validation_inputs, validation_targets,
                                  learning_rate, iterations=100, max_epoch=-1):
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
            self.train(training_inputs, training_targets,
                       learning_rate, iterations)
            oldest_error = previous_error
            previous_error = current_error
            output_value = self.mlpfwd(valid)
            current_error = (
                0.5 * np.sum((validation_targets - output_value)**2))

        logger.info("Stopped", current_error, previous_error, oldest_error)
        return current_error

    def train(self, inputs, targets, learning_rate, iterations, randomize=False):
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

        Returns
          Sum of squares error of the last network recall on the input data.

        """

        # Add the inputs that match the bias node
        training_dataset_count = np.shape(inputs)[0]
        inputs = np.concatenate(
            (inputs, -np.ones((training_dataset_count, 1))), axis=1)

        # Compute the initial order of input and target nodes so we can
        # randomize them if we so choose.
        node_order = list(range(training_dataset_count))

        # Create arrays to store update values for weight vectors.
        hidden_layer_updates = np.zeros((np.shape(self.hidden_weights)))
        output_layer_updates = np.zeros((np.shape(self.output_weights)))

        for iteration in range(iterations):
            self.outputs = self.mlpfwd(inputs)
            error = 0.5 * np.sum((self.outputs - targets)**2)
            if (np.mod(iteration, 100) == 0):
                logger.info("Iteration: ", iteration, " Error: ", error)

            # Compute the output layer error gradient for different activation functions.
            if self.output_type == OutputType.LINEAR:
                deltao = (self.outputs - targets) / training_dataset_count

            elif self.output_type == OutputType.LOGISTIC:
                deltao = self.beta * (self.outputs - targets) * \
                    self.outputs * (1.0 - self.outputs)

            elif self.output_type == OutputType.SOFTMAX:
                deltao = (self.outputs - targets) * (self.outputs *
                                                     (-self.outputs) + self.outputs) / training_dataset_count

            else:
                raise InvalidOutputTypeError(
                    'output_type not member of OutputType')

            # Compute the hidden layer error gradient for logistic activation function.
            deltah = self.hidden * self.beta * \
                (1.0 - self.hidden) * \
                (np.dot(deltao, np.transpose(self.output_weights)))

            # Use error gradients to compute weight update values.
            # We're incorporating the previous weight changes to give them some
            # "momentum." This is done to help prevent the algorithm from
            # becoming stuck in local optima.
            hidden_layer_updates = learning_rate * (np.dot(np.transpose(inputs),
                                                           deltah[:, :-1])) + self.momentum * hidden_layer_updates

            output_layer_updates = learning_rate * (np.dot(np.transpose(self.hidden),
                                                           deltao)) + self.momentum * output_layer_updates

            # Apply weight update values to hidden and output layer weights.
            self.hidden_weights -= hidden_layer_updates
            self.output_weights -= output_layer_updates

            if randomize:
                # Randomize order of input vector and update target vector correspondingly.
                np.random.shuffle(node_order)
                inputs = inputs[node_order, :]
                targets = targets[node_order, :]

        return error

    def mlpfwd(self, inputs):
        """ Run the network forward """

        self.hidden = np.dot(inputs, self.hidden_weights)
        self.hidden = 1.0 / (1.0 + np.exp(-self.beta * self.hidden))
        self.hidden = np.concatenate(
            (self.hidden, -np.ones((np.shape(inputs)[0], 1))), axis=1)

        outputs = np.dot(self.hidden, self.output_weights)

        # Different types of output neurons
        if self.output_type == OutputType.LINEAR:
            return outputs

        elif self.output_type == OutputType.LOGISTIC:
            return 1.0 / (1.0 + np.exp(-self.beta * outputs))

        elif self.output_type == OutputType.SOFTMAX:
            normalisers = np.sum(np.exp(outputs), axis=1) * \
                np.ones((1, np.shape(outputs)[0]))

            return np.transpose(np.transpose(np.exp(outputs)) / normalisers)

        else:
            raise InvalidOutputTypeError(
                'output_type not member of OutputType')

    def confmat(self, inputs, targets):
        """Confusion matrix"""

        # Add the inputs that match the bias node
        inputs = np.concatenate(
            (inputs, -np.ones((np.shape(inputs)[0], 1))), axis=1)

        outputs = self.mlpfwd(inputs)
        nclasses = np.shape(targets)[1]
        if nclasses == 1:
            nclasses = 2
            outputs = np.where(outputs > 0.5, 1, 0)

        else:
            # 1-of-N encoding
            outputs = np.argmax(outputs, 1)
            targets = np.argmax(targets, 1)

        cm = np.zeros((nclasses, nclasses))
        for i in range(nclasses):
            for j in range(nclasses):
                cm[i, j] = np.sum(np.where(outputs == i, 1, 0)
                                  * np.where(targets == j, 1, 0))

        logger.info("Confusion matrix is:")
        logger.info(str(cm))
        logger.info("Percentage Correct: ", np.trace(cm) / np.sum(cm) * 100)
        return cm


class InvalidOutputTypeError(Exception):
    """
    Indicates that an invalid output type was specified.
    """
