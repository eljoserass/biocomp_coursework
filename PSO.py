from ANN import ANN, create_layer

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def get_particles_accuracy(particles_position, ann: ANN, take_sample = True):
    # print("hello") 
    particles_accuracy = np.array([])
    for i in range(len(particles_position)):
        ann.fill_weights(particle=particles_position[i])
        particles_accuracy = np.append(particles_accuracy, ann.get_accuracy()) # cambiar con get_accuracy_train()
    return particles_accuracy

def get_particles_cost(particles_position, ann: ANN, take_sample = True):
    particles_cost = np.array([])
    if take_sample == True:
        ann.take_samples()  
    for i in range(len(particles_position)):
        ann.fill_weights(particle=particles_position[i])
        ann.forward_pass()
        particles_cost = np.append(particles_cost, ann.get_cost())
    return particles_cost

def update_particles_velocity(particles_position, particles_velocity ,particles_inertia, particles_position_pbest, particle_position_gbest, c1, c2):
    # TODO see informants stratergies
    new_particle_velocity = (
        particles_inertia * particles_velocity + 
        c1 * np.random.rand(1,)[0] * (particles_position_pbest - particles_position) + 
        c2 * np.random.rand(1,)[0] * (particle_position_gbest - particles_position)
    )
    return new_particle_velocity

def update_particle_position(particles_position, particles_velocity):
    return particles_position + particles_velocity

def update_particle_velocity_position(particles_position, particles_velocity ,particles_inertia, particles_position_pbest, particle_position_gbest, c1, c2):
    particles_velocity = update_particles_velocity(particles_position=particles_position,
                                                    particles_velocity=particles_velocity,
                                                    particles_inertia=particles_inertia,
                                                    particles_position_pbest=particles_position_pbest,
                                                    particle_position_gbest=particle_position_gbest,
                                                    c1=c1,
                                                    c2=c2
                                                   )
    return particles_position + particles_velocity, particles_velocity

def get_particle_cost(particle_position, ann):
    ann.fill_weights(particle=particle_position)
    ann.forward_pass()
    return ann.cost

def get_particle_accuracy(particle_position, ann):
    ann.fill_weights(particle=particle_position)
    ann.forward_pass()
    return ann.accuracy

def get_variable_parameters(kwargs_pso):
    
    return

def pso_min_cost(num_particles: int, ann: ANN, max_iter: int, **kwargs):
    np.random.seed(kwargs.get("seed"))
    particles_inertia = np.random.rand(num_particles, ) if kwargs.get("particles_inertia") is None\
                        else np.full(num_particles, kwargs.get("particles_inertia"))
    c1 = np.random.rand(1,)[0] if  kwargs.get("c1") is None else kwargs.get("c1")
    c2 = np.random.rand(1,)[0] if  kwargs.get("c2") is None else kwargs.get("c2")
    particles_position = np.random.rand(num_particles, ann.get_total_parameters())
    particles_cost = np.full(num_particles, 1)
    particles_velocity = np.random.rand(num_particles, ann.get_total_parameters())
    particles_position_pbest = np.copy(particles_position)
    particles_position_pbest_cost =  np.copy(particles_cost)
    particle_position_gbest = particles_position[np.argmin(particles_cost)]
    particle_position_gbest_cost = np.min(particles_cost)
    # c1 = np.random.rand(1,)[0] # maybe not random
    # c2 = np.random.rand(1,)[0] # maybe not random
    # for some reason if the 2 lines above are uncommented it gives better results
    
    for iteration in range(max_iter):
        if iteration % 10 == 0:
            print (f"\t- Iteration: {iteration}")
        while not ann.finished_batch:
            particles_cost = get_particles_cost(particles_position=particles_position, ann=ann)
            for i in range(num_particles):
                if particles_cost[i] < particles_position_pbest_cost[i]:
                    particles_position_pbest[i] = np.copy(particles_position[i])
                    particles_position_pbest_cost[i] = particles_cost[i]

                if  particles_cost[i] < particle_position_gbest_cost:
                    particle_position_gbest = np.copy(particles_position[i])
                    particle_position_gbest_cost = particles_cost[i]
                    particle_gbest_inertia = particles_inertia[i]
 
                particles_velocity[i] = update_particles_velocity(particles_position=particles_position[i],
                                                        particles_velocity=particles_velocity[i],
                                                        particles_inertia=particles_inertia[i],
                                                        particles_position_pbest=particles_position_pbest[i],
                                                        particle_position_gbest=particle_position_gbest,
                                                        c1=c1,  
                                                        c2=c2)
                particles_position[i] = particles_position[i] + particles_velocity[i]
        ann.finished_batch = False

    return {
            "gbest_position": particle_position_gbest, 
            "gbest_cost": particle_position_gbest_cost,
            "gbest_inertia": particle_gbest_inertia,
            "c1": c1,
            "c2": c2
            }



def pso_max_accuracy(num_particles: int, ann: ANN, max_iter: int, **kwargs):
    np.random.seed(kwargs.get("seed"))
    particles_inertia = np.random.rand(num_particles, ) if kwargs.get("particles_inertia") is None\
                        else np.full(num_particles, kwargs.get("particles_inertia"))
    c1 = np.random.rand(1,)[0] if  kwargs.get("c1") is None else kwargs.get("c1")
    c2 = np.random.rand(1,)[0] if  kwargs.get("c2") is None else kwargs.get("c2")

    particles_position = np.random.rand(num_particles, ann.get_total_parameters())
    
    particles_cost = np.full(num_particles, 1)
    particles_accuracy = np.full(num_particles, 0)
    
    particles_velocity = np.random.rand(num_particles, ann.get_total_parameters())
    
    particles_position_pbest =  np.copy(particles_position)
    particles_position_pbest_accuracy = np.copy(particles_accuracy)
    
    particle_position_gbest = particles_position[np.argmax(particles_accuracy)]
    particle_position_gbest_cost = np.min(particles_cost)
    particle_position_gbest_accuracy = np.max(particles_position_pbest_accuracy)
    
    particle_gbest_inertia = None

    for iteration in range(max_iter):
        if iteration % 10 == 0:
            print (f"\t- Iteration: {iteration}")
        while not ann.finished_batch:
            particles_accuracy = get_particles_accuracy(particles_position=particles_position, ann=ann)
            particles_cost = get_particles_cost(particles_position=particles_position, ann=ann, take_sample=False)
            # quiero guardar el costo pero si la descomento tarda mucho
            for i in range(num_particles):
                if particles_accuracy[i] > particles_position_pbest_accuracy[i]:
                    particles_position_pbest[i] = particles_position[i]
                
                if  particles_accuracy[i] > particle_position_gbest_accuracy:
                    particle_position_gbest = np.copy(particles_position[i])
                    particle_position_gbest_cost = particles_cost[i]
                    particle_position_gbest_accuracy = particles_accuracy[i]
                    particle_gbest_inertia = particles_inertia[i]
                    
                
                particles_velocity[i] = update_particles_velocity(particles_position=particles_position[i],
                                                        particles_velocity=particles_velocity[i],
                                                        particles_inertia=particles_inertia[i],
                                                        particles_position_pbest=particles_position_pbest[i],
                                                        particle_position_gbest=particle_position_gbest,
                                                        c1=c1,
                                                        c2=c2)
                particles_position[i] = particles_position[i] + particles_velocity[i]
        ann.finished_batch = False

    return {
            "gbest_position": particle_position_gbest, 
            "gbest_cost": particle_position_gbest_cost,
            "gbest_inertia": particle_gbest_inertia,
            "c1": c1,
            "c2": c2
            }
