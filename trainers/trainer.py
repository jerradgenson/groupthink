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

from abc import ABC, abstractmethod
from copy import copy

import numpy as np


class Trainer(ABC):
    """
    Abstract base class for all Trainer classes.
    Subclasses that override constructor must call base class constructor.

    Args
      iterations: Number of iterations to run the Trainer on the Learner per call.

    """

    def __init__(self, iterations):
        self.iterations = iterations

    @abstractmethod
    def _train(self, learner, inputs, targets):
        """
        Run the training algorithm on the given Learner instance.

        Args
          learner: An instance of learners.Learner.
          inputs: Training inputs to the network as a numpy array of arrays,
                  where each inner array is one set of inputs.
          targets: Target outputs for the network as a numpy array of arrays,
                   where each inner array is one set of target outputs. Target
                   arrays must match the order of input arrays.

        Returns
          Sum of squares error of the last recall on the input data.

        """

    def train(self, learner, inputs, targets, *args, **kwargs):
        self._train(learner, inputs, targets, *args, **kwargs)
        outputs = learner._recall(inputs)
        error = self.calculate_error(outputs, targets)
        return error

    train.__doc__ = _train.__doc__

    @staticmethod
    def calculate_error(outputs, targets):
        """
        Calculate the sum-of-squares error given sets of outputs and targets.

        Args
          outputs: A numpy array of arrays representing a learner's outputs,
                   where each inner array corresponds to an inner array in the
                   inputs.
          targets: A numpy array of arrays representing a learner's targets,
                   values to compare against the learner's outputs to calculate
                   the error. targets must be the same shape as outputs.

        Returns
          A floating-point value indicating the sum-of-squares error.

        """

        return 0.5 * np.sum((outputs - targets) ** 2)

    @staticmethod
    def generate_population(learner, population_size):
        """
        Generate a population of learners from an initial learner. Target Learner
        must provide a randomize() method to randomize its internal state and an
        update(new_state) method to update its internal state. The target
        Learner instance is added to the population, but is not modified.

        Args
          learner: An instance of learners.learner.Learner.
          population_size: The size of the population to generate.

        Yields
          A population of Learners derived from the target learner.

        """

        for index in range(population_size):
            if index == 0:
                yield learner

            else:
                new_learner = copy(learner)
                new_learner.update(learner.randomize())
                yield new_learner
