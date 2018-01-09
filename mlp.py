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
    """ A Multi-Layer Perceptron"""

    def __init__(self, input_node_count, hidden_node_count, output_node_count,
                 beta=1, momentum=0.9, output_type=OutputType.LOGISTIC):

        self.beta = beta
        self.momentum = momentum
        self.output_type = output_type

        # Initialise network
        self.hidden_nodes = (np.random.rand(
            input_node_count + 1, hidden_node_count) - 0.5) * 2 / np.sqrt(input_node_count)

        self.output_nodes = (np.random.rand(hidden_node_count + 1,
                                            output_node_count) - 0.5) * 2 / np.sqrt(hidden_node_count)

    def train_with_early_stopping(self, training_inputs, training_targets,
                                  validation_inputs, validation_targets,
                                  learning_rate, iterations=100, max_epoch=-1):

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

    def train(self, inputs, targets, learning_rate, iterations):

        # Add the inputs that match the bias node
        training_dataset_count = np.shape(inputs)[0]
        inputs = np.concatenate(
            (inputs, -np.ones((training_dataset_count, 1))), axis=1)
        change = list(range(training_dataset_count))
        updatew1 = np.zeros((np.shape(self.hidden_nodes)))
        updatew2 = np.zeros((np.shape(self.output_nodes)))
        for n in range(iterations):
            self.outputs = self.mlpfwd(inputs)
            error = 0.5 * np.sum((self.outputs - targets)**2)
            if (np.mod(n, 100) == 0):
                logger.info("Iteration: ", n, " Error: ", error)

            # Different types of output neurons
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

            deltah = self.hidden * self.beta * \
                (1.0 - self.hidden) * \
                (np.dot(deltao, np.transpose(self.output_nodes)))

            updatew1 = learning_rate * (np.dot(np.transpose(inputs),
                                               deltah[:, :-1])) + self.momentum * updatew1

            updatew2 = learning_rate * (np.dot(np.transpose(self.hidden),
                                               deltao)) + self.momentum * updatew2

            self.hidden_nodes -= updatew1
            self.output_nodes -= updatew2

            # Randomise order of inputs (not necessary for matrix-based calculation)
            # np.random.shuffle(change)
            #inputs = inputs[change,:]
            #targets = targets[change,:]

    def mlpfwd(self, inputs):
        """ Run the network forward """

        self.hidden = np.dot(inputs, self.hidden_nodes)
        self.hidden = 1.0 / (1.0 + np.exp(-self.beta * self.hidden))
        self.hidden = np.concatenate(
            (self.hidden, -np.ones((np.shape(inputs)[0], 1))), axis=1)

        outputs = np.dot(self.hidden, self.output_nodes)

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
