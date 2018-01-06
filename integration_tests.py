"""
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

Inspired by code from "Machine Learning: An Algorithmic Perspective" by
Dr. Stephen Marsland (http://stephenmonika.net).

"""

import unittest
import numpy as np
import mlp


class TestMLP(unittest.TestCase):

    def test_logical_and(self):
        and_data = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 1]])
        neural_net = mlp.MultilayerPerceptron(2, 2, 1)
        neural_net.mlptrain(and_data[:, 0:2], and_data[:, 2:3], 0.25, 1001)
        confusion_matrix = neural_net.confmat(
            and_data[:, 0:2], and_data[:, 2:3])

        expected = np.array([[3., 0.], [0., 1.]])
        self.assertTrue((confusion_matrix == expected).all())

    def test_logical_xor(self):
        xor_data = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]])
        neural_net = mlp.MultilayerPerceptron(2, 2, 1,
                                              output_type=mlp.OutputType.LOGISTIC)

        neural_net.mlptrain(xor_data[:, 0:2], xor_data[:, 2:3], 0.25, 5001)
        neural_net.confmat(xor_data[:, 0:2], xor_data[:, 2:3])
        confusion_matrix = neural_net.confmat(
            xor_data[:, 0:2], xor_data[:, 2:3])

        expected = np.array([[2., 0.], [0., 2.]])
        self.assertTrue((confusion_matrix == expected).all())


if __name__ == '__main__':
    unittest.main()
