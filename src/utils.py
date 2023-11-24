import numpy as np
import os
import pathlib
import pandas as pd

def standard_deviation_func(accuracity : list ):
    mean = np.mean(accuracity)
    result = 0
    for x in accuracity:
        result +=  (x - mean) **2
    return result / len(accuracity)


def count_true(y_true, y_pred):
    correct_predictions = 0
    for i in range(len(y_true)):
        if y_pred[i] == y_true[i]:
            correct_predictions += 1
    return correct_predictions

def write_evolution(standard_deviation, particles_position, particles_accuracy, particles_cost, particles_velocity, iteration, experiment_id, step = 10):
    if iteration % step != 0 or experiment_id == None:
        return

    results = {
        "standard_deviation": [],
        "position": [],
        "accuracy": [],
        "cost": [],
        "velocity": [],
        "iteration": [],
        "experiment_id": []
    }

    for i in range(len(particles_position)):
        results["position"].append(particles_position[i].tolist())
        results["accuracy"].append(particles_accuracy[i].tolist())
        results["cost"].append(particles_cost[i].tolist())
        results["velocity"].append(particles_velocity[i].tolist())
        results["standard_deviation"].append(standard_deviation)
        results["iteration"].append(iteration)
        results["experiment_id"].append(experiment_id)
    
    if not os.path.exists(f"./evolution_viz/{experiment_id}"):
        os.mkdir(f"./evolution_viz/{experiment_id}")
    df = pd.DataFrame(results)
    csvfile = pathlib.Path(f"./evolution_viz/{experiment_id}/{iteration}.csv")
    df.to_csv(f"./evolution_viz/{experiment_id}/{iteration}.csv", mode='a', index=False, header=not csvfile.exists())