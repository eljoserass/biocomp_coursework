from src.ANN import ANN
import numpy as np
from src.utils import standard_deviation_func, write_evolution

class PSO:
    def __init__(self, Ann: ANN):
        self.ann = Ann

    #update the particles accuracy
    def update_particles_accuracy(self):
        self.particles_accuracy = np.array([])
        for i in range(len(self.particles_position)):
            self.ann.fill_weights(particle=self.particles_position[i])
            self.particles_accuracy = np.append(self.particles_accuracy, self.ann.get_accuracy())

    #update paticles cost
    def update_particles_cost(self, take_sample = True):
        self.particles_cost = np.array([])
        if take_sample == True:
            self.ann.take_samples()  
        for i in range(len(self.particles_position)):
            self.ann.fill_weights(particle=self.particles_position[i])
            self.ann.forward_pass()
            self.particles_cost = np.append(self.particles_cost, self.ann.get_cost())

    #update particles velocity
    def update_particles_velocity(self, index):
        new_particle_velocity = (
            self.particles_inertia[index] * self.particles_velocity[index] + 
            self.c1 * np.random.rand(1,)[0] * (self.particles_position_pbest[index] - self.particles_position[index]) + 
            self.c2 * np.random.rand(1,)[0] * (self.particle_position_gbest -self.particles_position[index])
        )
        return new_particle_velocity


    #init variales to calculate the pso with cost as a learning
    def init_pso_min_cost(self, num_particles: int):
        self.particles_position = np.random.rand(num_particles, self.ann.get_total_parameters())
        self.particles_cost = np.full(num_particles, 1)
        self.particles_velocity = np.random.rand(num_particles, self.ann.get_total_parameters())
        self.particles_position_pbest = np.copy(self.particles_position)
        self.particles_position_pbest_cost =  np.copy(self.particles_cost)
        self.particle_position_gbest = self.particles_position[np.argmin(self.particles_cost)]
        self.particle_position_gbest_cost = np.min(self.particles_cost)
        self.particle_gbest_inertia = None
        self.standard_deviation = None

    #run pso with  min cost as a learning
    def pso_min_cost(self, num_particles: int, max_iter: int, **kwargs):
        #init variables
        np.random.seed(kwargs.get("seed"))
        self.particles_inertia = np.random.rand(num_particles, ) if kwargs.get("particles_inertia") is None\
                            else np.full(num_particles, kwargs.get("particles_inertia"))
        self.c1 = np.random.rand(1,)[0] if  kwargs.get("c1") is None else kwargs.get("c1")
        self.c2 = np.random.rand(1,)[0] if  kwargs.get("c2") is None else kwargs.get("c2")
        self.init_pso_min_cost(num_particles=num_particles)
        
        #for each epoch
        for iteration in range(max_iter):
            #print the numbers of iterations done each 10
            if iteration % 10 == 0:
                print (f"\t- Iteration: {iteration}")
            #for each batch
            while not self.ann.finished_batch:
                #update the particle cost
                self.update_particles_cost()
                #for each particle
                for i in range(num_particles):
                    #if a particle has found the lowest personal cost 
                    if self.particles_cost[i] < self.particles_position_pbest_cost[i]:
                        self.particles_position_pbest[i] = np.copy(self.particles_position[i])
                        self.particles_position_pbest_cost[i] = self.particles_cost[i]
                    # if a particle has found the lowest global cost
                    if  self.particles_cost[i] < self.particle_position_gbest_cost:
                        self.particle_position_gbest = np.copy(self.particles_position[i])
                        self.particle_position_gbest_cost = self.particles_cost[i]
                        self.particle_gbest_inertia = self.particles_inertia[i]
                    #update velocity
                    self.particles_velocity[i] = self.update_particles_velocity(i)
                    #updating the position of the particle to a better position 
                    self.particles_position[i] = self.particles_position[i] + self.particles_velocity[i]
            self.ann.finished_batch = False

        return {
                "gbest_position": self.particle_position_gbest, 
                "gbest_cost": self.particle_position_gbest_cost,
                "gbest_inertia": self.particle_gbest_inertia,
                "c1": self.c1,
                "c2": self.c2
                }

    #init variales to calculate the pso with accuracy as a learning
    def init_pso_max_accuracy(self, num_particles: int):
        self.particles_position = np.random.rand(num_particles, self.ann.get_total_parameters())
        
        self.particles_cost = np.full(num_particles, 1)
        self.particles_accuracy = np.full(num_particles, 0)
        
        self.particles_velocity = np.random.rand(num_particles, self.ann.get_total_parameters())
        
        self.particles_position_pbest =  np.copy(self.particles_position)
        self.particles_position_pbest_accuracy = np.copy(self.particles_accuracy)
        
        self.particle_position_gbest = self.particles_position[np.argmax(self.particles_accuracy)]
        self.particle_position_gbest_cost = np.min(self.particles_cost)
        self.particle_position_gbest_accuracy = np.max(self.particles_position_pbest_accuracy)
        
        self.particle_gbest_inertia = None

    #run pso with  min accuracy as a learning
    def pso_max_accuracy(self, num_particles: int, max_iter: int, **kwargs):
        #init variables
        np.random.seed(kwargs.get("seed"))
        self.particles_inertia = np.random.rand(num_particles, ) if kwargs.get("particles_inertia") is None\
                            else np.full(num_particles, kwargs.get("particles_inertia"))
        self.c1 = np.random.rand(1,)[0] if  kwargs.get("c1") is None else kwargs.get("c1")
        self.c2 = np.random.rand(1,)[0] if  kwargs.get("c2") is None else kwargs.get("c2")
        self.init_pso_max_accuracy(num_particles=num_particles)
        
        #for each epoch
        for iteration in range(max_iter):
            #print the numbers of iterations done each 10
            if iteration % 10 == 0:
                self.standard_deviation = standard_deviation_func(self.particles_accuracy)
                print (f"\t- Iteration: {iteration}")
            #for each batch
            while not self.ann.finished_batch:
                #update the particle cost
                self.update_particles_accuracy()
                self.update_particles_cost(take_sample=False)  #note: we just get the cost to annotate it later 
               #for each particle
                for i in range(num_particles):
                    if self.particles_accuracy[i] > self.particles_position_pbest_accuracy[i]:
                        self.particles_position_pbest[i] = self.particles_position[i]
                    
                    if  self.particles_accuracy[i] > self.particle_position_gbest_accuracy:
                        self.particle_position_gbest = np.copy(self.particles_position[i])
                        self.particle_position_gbest_cost = self.particles_cost[i]
                        self.particle_position_gbest_accuracy = self.particles_accuracy[i]
                        self.particle_gbest_inertia = self.particles_inertia[i]                       
                    #update velocity
                    self.particles_velocity[i] = self.update_particles_velocity(i)
                    #updating the position of the particle to a better position 
                    
                    self.particles_position[i] = self.particles_position[i] + self.particles_velocity[i]
                    #wirte the evolution in folder evolution_viz
                    write_evolution(
                        standard_deviation=self.standard_deviation,
                        particles_position=self.particles_position,
                        particles_accuracy=self.particles_accuracy,
                        particles_cost=self.particles_cost,
                        particles_velocity=self.particles_velocity,
                        iteration=iteration,
                        step=1,
                        experiment_id=kwargs.get("experiment_id")
                    )
            self.ann.finished_batch = False

        return {
                "gbest_position": self.particle_position_gbest, 
                "gbest_cost": self.particle_position_gbest_cost,
                "gbest_inertia": self.particle_gbest_inertia,
                "c1": self.c1,
                "c2": self.c2
                }
