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

    def earlystopping(self, inputs, targets, valid, validtargets, eta, niterations=100):

        valid = np.concatenate(
            (valid, -np.ones((np.shape(valid)[0], 1))), axis=1)

        old_val_error1 = 100002
        old_val_error2 = 100001
        new_val_error = 100000
        count = 0
        while (((old_val_error1 - new_val_error) > 0.001) or ((old_val_error2 - old_val_error1) > 0.001)):
            count += 1
            logger.info(count)
            self.mlptrain(inputs, targets, eta, niterations)
            old_val_error2 = old_val_error1
            old_val_error1 = new_val_error
            validout = self.mlpfwd(valid)
            new_val_error = 0.5 * np.sum((validtargets - validout)**2)

        logger.info("Stopped", new_val_error, old_val_error1, old_val_error2)
        return new_val_error

    def mlptrain(self, inputs, targets, eta, niterations):
        """ Train the thing """
        # Add the inputs that match the bias node
        training_dataset_count = np.shape(inputs)[0]
        inputs = np.concatenate(
            (inputs, -np.ones((training_dataset_count, 1))), axis=1)
        change = list(range(training_dataset_count))
        updatew1 = np.zeros((np.shape(self.hidden_nodes)))
        updatew2 = np.zeros((np.shape(self.output_nodes)))
        for n in range(niterations):
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

            updatew1 = eta * (np.dot(np.transpose(inputs),
                                     deltah[:, :-1])) + self.momentum * updatew1

            updatew2 = eta * (np.dot(np.transpose(self.hidden),
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
