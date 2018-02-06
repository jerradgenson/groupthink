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

import unittest
import sys

import numpy as np

sys.path.append('..')
from learners import mlp
from preprocessing import normalize


class TestMLP(unittest.TestCase):
    AND_DATA = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 1]])
    XOR_DATA = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]])

    def test_logical_and(self):
        and_data = self.AND_DATA
        neural_net = mlp.MultilayerPerceptron((2, 2, 1), (False, True))
        neural_net.train(and_data[:, 0:2], and_data[:, 2:], 0.25, 1001)
        confusion_matrix = neural_net.generate_confusion_matrix(
            and_data[:, 0:2], and_data[:, 2:3])

        expected = np.array([[3., 0.], [0., 1.]])
        self.assertTrue((confusion_matrix == expected).all())

        output = neural_net.recall(and_data[:, 0:2])
        self.assertEqual(output, [False, False, False, True])

    def test_false_bias(self):
        and_data = self.AND_DATA
        neural_net = mlp.MultilayerPerceptron((2, 2, 1), (False, True), bias=False)
        training_inputs = and_data[:, 0:2]
        training_targets = and_data[:, 2:]
        neural_net.train_with_early_stopping(training_inputs, training_targets,
                                             training_inputs, training_targets,
                                             0.25, 1001)

    def test_logical_xor(self):
        xor_data = self.XOR_DATA
        neural_net = mlp.MultilayerPerceptron((2, 2, 1), (False, True))

        neural_net.train(xor_data[:, 0:2], xor_data[:, 2:], 0.25, 5001)
        confusion_matrix = neural_net.generate_confusion_matrix(xor_data[:, 0:2],
                                                                xor_data[:, 2:3])

        expected = np.array([[2., 0.], [0., 2.]])
        self.assertTrue((confusion_matrix == expected).all())

        output = neural_net.recall(xor_data[:, 0:2])
        self.assertEqual(output, [False, True, True, False])

    def test_sine_regression(self):
        errors = []
        for iteration in range(10):
            input_data = np.ones((1, 40)) * np.linspace(0, 1, 40)
            target_data = (np.sin(2 * np.pi * input_data) +
                           np.cos(4 * np.pi * input_data) +
                           np.random.randn(40) * 0.2)

            input_data = np.transpose(input_data)
            target_data = np.transpose(target_data)
            input_data = normalize(input_data)
            target_data = normalize(target_data)

            training_inputs = input_data[0::2, :]
            testing_inputs = input_data[1::4, :]
            validation_inputs = input_data[3::4, :]
            training_targets = target_data[0::2, :]
            testing_targets = target_data[1::4, :]
            validation_targets = target_data[3::4, :]

            neural_net = mlp.MultilayerPerceptron((1, 5, 4, 3, 1),
                                                  learner_type=mlp.LearnerType.REGRESSION)

            neural_net.train_with_early_stopping(training_inputs, training_targets,
                                                 validation_inputs, validation_targets,
                                                 0.25, 800)

            testing_outputs = neural_net.recall(testing_inputs)
            errors.append(0.5 * np.sum((testing_targets - testing_outputs)**2))

        average_error = np.median(errors)
        self.assertLessEqual(average_error, 0.5)


if __name__ == '__main__':    
    unittest.main()
