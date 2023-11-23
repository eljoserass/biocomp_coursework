from datetime import datetime
import pandas as pd
import numpy as np
from ANN import ANN, create_layer
from PSO import pso_min_cost, pso_max_accuracy

def get_time_str() -> str:
    now = datetime.now()
    return now.strftime("%d/%m/%Y-%H:%M:%S")

def architecture_to_str(architecture: list) -> str:
    result_str: str = f"n_layers:{len(architecture)}"
    for layer in architecture:
        result_str += ";"
        result_str += layer.function
        result_str += ":"
        result_str += layer.n_perceptrons

def write_experiments(results: dict,
                      db_path: str
                  ) -> pd.DataFrame:
    import pathlib
    df = pd.DataFrame(results)
    # TODO check if file is created and if header is created
    csvfile = pathlib.Path(db_path)
    df.to_csv(db_path, mode='a', index=False, header=not csvfile.exists())
    return df

def str_to_layers(architecture_str: str):
    layers_str = architecture_str.split(";")
    layers = []
    for layer in layers_str:
        layer_config = layer.split(":")
        layers.append(create_layer(function=layer_config[0], n_perceptrons=int(layer_config[1])))
    return layers
        
def get_output_activation(architecture_str: str):
    layers = architecture_str.split(";")
    return layers[-1].split(":")[0]
    

def run_experiments(X_train, X_test, y_train, y_test, config_path: str, session_name: str = "", db_path="experiments_db.csv", evolution_viz = False):
    config = pd.read_csv(config_path)
    results = {
        "particle_cost": [],
        "particle_inertia": [],
        "c1": [],
        "c2": [],
        "cost_fn": [],
        "accuracy": [],
        "correct_predictions": [],
        "particles": [],
        "iterations": [],
        "objective": [],
        "architecture": [],
        "batch_size": [],
        "seed": [],
        "session_name": [],
        "particle_pos": [],
        "time": [],
    }
    ann = None
    pso_args = None
    for index, row in config.iterrows():
        print (f"Running Experiment {index}")
        batch = None if np.isnan(row["batch_size"]) else int(row["batch_size"])
        ann = ANN(layers=str_to_layers(row["architecture"]), cost_fn=row["cost_fn"], Xdata=X_train.to_numpy(), Ydata=y_train.to_numpy(), batch=batch)
        
        pso_args = {
            "seed": None if np.isnan(row["seed"]) else int(row["seed"]),
            "c1": None if np.isnan(row["c1"]) else float(row["c1"]),
            "c2": None if np.isnan(row["c2"]) else float(row["c2"]),
            "particles_inertia": None if np.isnan(row["inertia"]) else float(row["inertia"]),
            "experiment_id": index if evolution_viz == True else None
        }
        
        if row["objective"] == "max_accuracy":
            pso_result = pso_max_accuracy(num_particles=row["particles"], max_iter=row["iterations"], ann=ann, **pso_args)
        if row["objective"] == "min_cost":
            pso_result = pso_min_cost(num_particles=row["particles"], max_iter=row["iterations"], ann=ann, **pso_args)
        ann = ANN(layers=str_to_layers(row["architecture"]), Xdata=X_test.to_numpy(), Ydata=y_test.to_numpy())
        ann.fill_weights(pso_result["gbest_position"])
        results["particle_cost"].append(pso_result["gbest_cost"])
        results["particle_inertia"].append(pso_result["gbest_inertia"])
        results["c1"].append(pso_result["c1"])
        results["c2"].append(pso_result["c2"])
        results["cost_fn"].append(row["cost_fn"])
        correct_predictions_print, accuracy = ann.get_accuracy(printable=True)
        results["accuracy"].append(accuracy)
        results["correct_predictions"].append(correct_predictions_print)
        results["particles"].append(row["particles"])
        results["iterations"].append(row["iterations"])
        results["objective"].append(row["objective"])
        results["architecture"].append(row["architecture"])
        results["batch_size"].append(batch)
        results["seed"].append(row["seed"])
        results["session_name"].append(session_name)
        results["particle_pos"].append(pso_result["gbest_position"].tolist())
        results["time"].append(get_time_str())
        print(f"* Best Accuracy {accuracy} | {correct_predictions_print}")
        print ("----------------")
    print ("\nExperiments Done!")
    write_experiments(db_path=db_path, results=results)


# def write_particle_cost_over_t