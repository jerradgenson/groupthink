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

from enum import Enum
from functools import partial
from abc import ABC, abstractmethod

import numpy as np


LearnerType = Enum(
    'LearnerType', 'CLASSIFICATION REGRESSION ONE_OF_N', module=__name__)


class Learner(ABC):
    """
    Abstract base class for all Learner classes.
    Child classes must call superclass constructor.

    Args
      trainer: An instance of trainers.Trainer. Specifies the algorithm used to
               train the Learner.
      classes: A sequence of classification names. This must be defined for
               ONE_OF_N learners. Defaults to None.
      learner_type: One of the values enumerated by LearnerType. Indicates
                    whether to construct a classification, regression, or
                    one_of_n learner. Defaults to classification.

    """

    def __init__(self, trainer, classes=None, learner_type=LearnerType.CLASSIFICATION):
        self.trainer = trainer
        self.train = partial(trainer.train, self)
        self.classes_of_outputs = None
        if learner_type == LearnerType.ONE_OF_N and classes:
            def classes_of_outputs(self, outputs):
                output_classes = []
                for output_index, output in enumerate(outputs):
                    maximum = np.amax(output)
                    class_index = np.where(output == maximum)
                    output_class = np.array(classes[class_index])
                    output_classes.append(output_class)

                return output_classes

            self.classes_of_outputs = classes_of_outputs

        elif learner_type == LearnerType.CLASSIFICATION and classes:
            # Create mapping from output value to class.
            increment = 1 / len(classes)
            bounds = [count * increment for count in range(len(classes))]

            def class_of_output(output):
                current_class = None
                for output_class, bound in zip(classes, bounds):
                    if output >= bound:
                        current_class = output_class

                    else:
                        break

                return current_class

            def classes_of_outputs(outputs):
                output_classes = []
                for index, output in enumerate(outputs):
                    output_class = class_of_output(output)
                    output_classes.append(output_class)

                return output_classes

            self.classes_of_outputs = classes_of_outputs

    @abstractmethod
    def _recall(self, inputs):
        """
        Args
          inputs: Input data to the Learner as a numpy array of arrays, where
                  each inner array is one set of inputs.

        Returns
          A numpy array of arrays representing the Learner's outputs, where each
          inner array corresponds to an inner array in the inputs.

        """

    def recall(self, inputs):
        outputs = self._recall(inputs)
        if self.classes_of_outputs:
            outputs = self.classes_of_outputs(outputs)

        return outputs

    recall.__doc__ = _recall.__doc__


class InvalidLearnerTypeError(Exception):
    """
    Indicates that an invalid output type was specified.
    """
