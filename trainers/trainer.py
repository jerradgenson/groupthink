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
        return self._train(learner, inputs, targets, *args, **kwargs)

    train.__doc__ = _train.__doc__
