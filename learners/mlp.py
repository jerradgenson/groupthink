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

    def __init__(self, input_node_count, hidden_node_count, classes=None,
                 beta=1, learner_type=LearnerType.CLASSIFICATION, bias=True,
                 second_hidden_layer_node_count=0):

        self.beta = beta
        self.learner_type = learner_type
        self.bias = bias
        if learner_type in (LearnerType.CLASSIFICATION, LearnerType.REGRESSION):
            output_node_count = 1

        elif learner_type == LearnerType.ONE_OF_N:
            output_node_count = len(classes)

        else:
            raise InvalidLearnerTypeError(
                'learner_type not member of LearnerType')

        # Initialise network
        self.hidden_weights1 = (np.random.rand(
            input_node_count + 1, hidden_node_count) - 0.5) * 2 / np.sqrt(input_node_count)

        if second_hidden_layer_node_count > 0:
            node_count = second_hidden_layer_node_count
            self.hidden_weights2 = (np.random.rand(hidden_node_count + 1,
                                                   node_count) - 0.5) * 2 / np.sqrt(hidden_node_count)

        else:
            node_count = hidden_node_count
            self.hidden_weights2 = None

        self.output_weights = (np.random.rand(node_count + 1,
                                              output_node_count) - 0.5) * 2 / np.sqrt(node_count)

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
        training_dataset_count = np.shape(inputs)[0]
        inputs = np.concatenate(
            (inputs, -np.ones((training_dataset_count, 1))), axis=1)

        # Compute the initial order of input and target nodes so we can
        # randomize them if we so choose.
        node_order = list(range(training_dataset_count))

        # Create arrays to store update values for weight vectors.
        hidden_layer_updates1 = np.zeros((np.shape(self.hidden_weights1)))
        output_layer_updates = np.zeros((np.shape(self.output_weights)))
        if self.hidden_weights2 is not None:
            hidden_layer_updates2 = np.zeros((np.shape(self.hidden_weights2)))

        for iteration in range(iterations):
            self.outputs = self.__recall(False, inputs)
            error = 0.5 * np.sum((self.outputs - targets)**2)
            if (np.mod(iteration, 100) == 0):
                logger.info("Iteration: ", iteration, " Error: ", error)

            # Compute the output layer error gradient for different activation functions.
            if self.learner_type == LearnerType.REGRESSION:
                deltao = (self.outputs - targets) / training_dataset_count

            elif self.learner_type == LearnerType.CLASSIFICATION:
                deltao = self.beta * (self.outputs - targets) * \
                    self.outputs * (1.0 - self.outputs)

            elif self.learner_type == LearnerType.ONE_OF_N:
                deltao = (self.outputs - targets) * (self.outputs *
                                                     (-self.outputs) + self.outputs) / training_dataset_count

            else:
                raise InvalidLearnerTypeError(
                    'learner_type not member of LearnerType')

            # Compute the hidden layer error gradient for logistic activation function.
            if self.hidden_weights2 is not None:
                deltah2 = self.hidden_outputs2 * self.beta * \
                    (1.0 - self.hidden_outputs2) * \
                    (np.dot(deltao, np.transpose(self.output_weights)))

                weights = self.hidden_weights2
                delta = deltah2

            else:
                weights = self.output_weights
                delta = deltao

            deltah1 = self.hidden_outputs1 * self.beta * \
                (1.0 - self.hidden_outputs1) * \
                (np.dot(delta, np.transpose(weights)))

            # Use error gradients to compute weight update values.
            # We're incorporating the previous weight changes to give them some
            # "momentum." This is done to help prevent the algorithm from
            # becoming stuck in local optima.
            hidden_layer_updates1 = learning_rate * (np.dot(np.transpose(inputs),
                                                            deltah1[:, :-1])) + momentum * hidden_layer_updates1

            if self.hidden_weights2 is not None:
                hidden_layer_updates2 = learning_rate * (np.dot(np.transpose(inputs),
                                                                deltah2[:, :-1])) + momentum * hidden_layer_updates2

            output_layer_updates = learning_rate * (np.dot(np.transpose(self.hidden_outputs1),
                                                           deltao)) + momentum * output_layer_updates

            # Apply weight update values to hidden and output layer weights.
            self.hidden_weights1 -= hidden_layer_updates1
            if self.hidden_weights2 is not None:
                self.hidden_weights2 -= hidden_layer_updates2

            self.output_weights -= output_layer_updates

            if randomize:
                # Randomize order of input vector and update target vector correspondingly.
                np.random.shuffle(node_order)
                inputs = inputs[node_order, :]
                targets = targets[node_order, :]

        return error

    def _logistic(self, weights):
        """
        Logistic activation function.
        
        Args
          weights: A numpy array of network layer outputs.

        Returns
          The value of the weights with logistic activation applied.

        """

        return 1.0 / (1.0 + np.exp(-self.beta * weights))

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

        if bias:
            # Add the inputs that match the bias node
            dataset_count = np.shape(inputs)[0]
            inputs = np.concatenate(
                (inputs, -np.ones((dataset_count, 1))), axis=1)

        # Compute hidden layer outputs from network inputs and hidden layer weights.
        self.hidden_outputs1 = np.dot(inputs, self.hidden_weights1)

        # Always use logistic activation function for the hidden layer.
        self.hidden_outputs1 = self._logistic(self.hidden_outputs1)

        # Concatenate bias node onto hidden layer outputs.
        self.hidden_outputs1 = np.concatenate(
            (self.hidden_outputs1, -np.ones((np.shape(inputs)[0], 1))), axis=1)

        if self.hidden_weights2 is not None:
            self.hidden_outputs2 = np.dot(self.hidden_outputs1,
                                          self.hidden_weights2)

            self.hidden_outputs2 = self._logistic(self.hidden_outputs2)

            self.hidden_outputs2 = np.concatenate(
                (self.hidden_outputs2, -np.ones((np.shape(inputs)[0], 1))), axis=1)

            hidden_outputs = self.hidden_outputs2

        else:
            hidden_outputs = self.hidden_outputs1

        # Compute network outputs from hidden layer outputs and output layer weights.
        network_outputs = np.dot(hidden_outputs, self.output_weights)
        if self.learner_type == LearnerType.REGRESSION:
            # Use linear activation (null activation function) for regression.
            return network_outputs

        elif self.learner_type == LearnerType.CLASSIFICATION:
            # Use logistic activation function for classification.
            return self._logistic(network_outputs)

        elif self.learner_type == LearnerType.ONE_OF_N:
            # Use soft-max activation function for 1-of-N classification.
            normalisers = np.sum(np.exp(network_outputs), axis=1) * \
                np.ones((1, np.shape(network_outputs)[0]))

            return np.transpose(np.transpose(np.exp(network_outputs)) / normalisers)

        else:
            raise InvalidLearnerTypeError(
                'learner_type not member of LearnerType')

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
