from ANN import ANN, create_layer

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

df_banknote = pd.read_csv("data_banknote_authentication.csv",  header=None)

X = df_banknote.iloc[:, [0,1,2,3]]
Y = df_banknote.iloc[:, [4]]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
print(type(X_train))
# ann = ANN(inputs=X.shape[1], layers=[Layer(function="sigmoid", n_perceptrons=1, n_inputs=X.shape[1], id=0, batch_size=X.shape[0])], Xdata=X, Ydata=Y)

# ann = ANN(inputs=X.shape[1], layers=[Layer(function="sigmoid", n_perceptrons=3, n_inputs=X.shape[1], id=0, batch_size=X.shape[0]),
#                             Layer(function="sigmoid", n_perceptrons=1, n_inputs=3, id=1, batch_size=X.shape[0])], Xdata=X, Ydata=Y)

architecture =[ create_layer("sigmoid", 2), create_layer("relu", 2), create_layer("tanh", 2)]
ann = ANN( layers=architecture, 
          Xdata=X_train.to_numpy(), Ydata=y_train.to_numpy())


ann.forward_pass()
ann.get_accuracy()

"""

particles = [[[w1,w2,w2], b1], [w3,w4,w5], b2, ....]


"""

def get_particles_cost(particles_position, ann: ANN):
    particles_cost = np.array([])
    
    for i in range(len(particles_position)):
        ann.fill_weights(particle=particles_position[i])
        ann.forward_pass()
        particles_cost = np.append(particles_cost, ann.get_cost())
    return particles_cost

def update_particles_velocity(particles_position, particles_velocity ,particles_inertia, particles_position_pbest, particle_position_gbest, c1, c2):
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

def get_particle_cost(particle_position):
    ann.fill_weights(particle=particle_position)
    ann.forward_pass()
    return ann.get_cost()


def pso(num_particles: int, ann: ANN, max_iter: int):
    particles_position = np.random.rand(num_particles, ann.get_total_parameters())
    particles_cost = get_particles_cost(particles_position=particles_position, ann=ann)
    particles_velocity = np.random.rand(num_particles, ann.get_total_parameters())
    particles_inertia = np.random.rand(num_particles, ) # maybe not random and not different for each
    # particles_inertia = np.full(num_particles, 0.7)
    particles_position_pbest = np.array(particles_position, copy=True)
    particles_position_pbest_cost =  get_particles_cost(particles_position=particles_position_pbest, ann=ann)
    particle_position_gbest = particles_position[np.argmin(particles_cost)]
    particle_position_gbest_cost = get_particle_cost(particle_position_gbest)
    c1 = np.random.rand(1,)[0] # maybe not random
    # c1 = 0.4
    c2 = np.random.rand(1,)[0] # maybe not random
    # c2 = 0.5

    for _ in range(max_iter):
        
        particles_cost = get_particles_cost(particles_position=particles_position, ann=ann) 
        particles_position_pbest_cost =  get_particles_cost(particles_position=particles_position_pbest, ann=ann)
        for i in range(num_particles):
            if particles_cost[i] < particles_position_pbest_cost[i]:
                particles_position_pbest[i] = particles_position[i]
            
            if  particles_cost[i] < particle_position_gbest_cost:
                particle_position_gbest = np.copy(particles_position[i])
                particle_position_gbest_cost = particles_cost[i]
            
            
            particles_velocity[i] = update_particles_velocity(particles_position=particles_position[i],
                                                    particles_velocity=particles_velocity[i],
                                                    particles_inertia=particles_inertia[i],
                                                    particles_position_pbest=particles_position_pbest[i],
                                                    particle_position_gbest=particle_position_gbest,
                                                    c1=c1,
                                                    c2=c2)
            particles_position[i] = particles_position[i] + particles_velocity[i]
    return particle_position_gbest, particle_position_gbest_cost

        
    
# print ("pre call")
position, cost= pso(num_particles=50, ann=ann, max_iter=200)
testing_ANN =  ANN(architecture, Xdata=X_test.to_numpy(), Ydata=y_test.to_numpy())
testing_ANN.fill_weights(position)
testing_ANN.forward_pass()
print("get accuracy", testing_ANN.get_accuracy())
print (f"position {position}   cost {cost}")
# print ("postcall")

