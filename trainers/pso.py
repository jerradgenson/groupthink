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

import numpy as np

from .trainer import Trainer


class ParticleSwarmOptimization(Trainer):
    """
    A generic PSO Trainer for training abitrary Learners.

    """

    def __init__(self, iterations, population_size, maximum_velocity,
                 learning_factor1=2, learning_factor2=2):
        
        self.population_size = population_size
        self.maximum_velocity = maximum_velocity
        self.learning_factor1 = learning_factor1
        self.learning_factor2 = learning_factor2
        self._gbest = None
        super().__init__(iterations)

    def _train(self, learner, inputs, targets):
        if not hasattr(learner, 'population'):
            # Learner doesn't have a population associated with it yet, so we
            # need to create one.
            learner.population = self.initialize_population(learner,
                                                            self.population_size)

        for iteration in range(self.iterations):
            if iteration > 0:
                # Do this first, but not on the first loop so we don't end up
                # doing these calculations needlessly on the last loop.
                for particle in learner.population:
                    # Calculate velocity and position for each particle.
                    if hasattr(particle, 'velocity'):
                        v = particle.velocity

                    else:
                        # Particle doesn't have a velocity, so we generate one.
                        v = np.zeros(particle.value)

                    # Rename variables so they are nicer to read in equation form.
                    c1 = self.learning_factor1
                    c2 = self.learning_factor2
                    rand = np.random.rand
                    pbest = particle.pbest.value
                    gbest = self.gbest.value
                    present = particle.value
                    vmax = self.maximum_velocity

                    # Calculate new particle velocity.
                    v = v + c1 * rand() * (pbest - present) + c2 * rand() * (gbest - present)

                    # Clamp velocities on each dimension to maximum velocity.
                    v = v.clip(-vmax, vmax)

                    # Calculate new particle position.
                    present = present + v
                    particle.value = present

            for particle in learner.population:
                # Calculate best personal and global fitness for each particle.
                learner.update(particle.value)
                particle.fitness = self.calculate_error(learner, inputs, targets)
                if not hasattr(particle, 'pbest') or particle.fitness > particle.pbest.fitness:
                    # Particle doesn't have a personal best fitness value yet or
                    # its current fitness exceeds pbest.
                    particle.pbest = particle

                if not self.gbest or particle.fitness > self.gbest.fitness:
                    # Global best fitness value doesn't exist yet or particle's
                    # current fitness exceeds gbest.
                    self.gbest = particle

        learner.update(self.gbest)
