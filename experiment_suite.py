from datetime import datetime
import pandas as pd
from ANN import ANN, create_layer
from PSO import pso

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
        

# def write_experiment(particle_pos:list,
#                   particle_cost:float,
#                   cost_fn:str,
#                   accuracy:float,
#                   particles: int, 
#                   iterations: int, 
#                   architecture: list
#                   ) -> pd.DataFrame:
#     df = pd.DataFrame()
#     df["particle_pos"] = particle_pos
#     df["particle_cost"] = particle_cost
#     df["cost_fn"] = cost_fn
#     df["accuracy"] = accuracy
#     df["particles"] = particles
#     df["iterations"] = iterations
#     df["architecture"] = architecture
#     df["time"] = get_time_str()
    
#     return df



def write_experiments(results: dict,
                      db_path: str
                  ) -> pd.DataFrame:
    df = pd.DataFrame(results)
    df.to_csv(db_path, mode='a', index=False, header=False)
    return df

def str_to_layers(architecture_str: str):
    layers_str = architecture_to_str.split(";")
    layers = []
    for layer in layers_str:
        layer_config = layer.split(":")
        layers.append(create_layer(function=layer_config[0], n_perceptrons=layer_config[1]))
    return layers
        
def get_output_activation(architecture_str: str):
    layers = architecture_str.split(";")
    return architecture_str[len(layers) - 1].split(":")[1]
    

def run_experiments(X_train, X_test, y_train, y_test, config_path: str, session_name: str = ""):
    config = pd.read_csv(config_path)
    results = {
        "particle_pos": [],
        "particle_cost": [],
        "cost_fn": [],
        "accuracy": [],
        "particles": [],
        "iterations": [],
        "architecture": [],
        "seed": [],
        "session_name": [],
        "time": []
    }
    ann = None
    
    for index, row in config.iterrows():
        print (f"Running Experiment {index}")
        ann = ANN(layers=row["architecture"], cost_fn=row["cost_fn"], Xdata=X_train.to_numpy(), Ydata=y_train.to_numpy())

        if row["seed"] == "":
            position, cost = pso(num_particles=row["particles"], max_iter=row["max_iter"], ann=ann)
        else:
            position, cost = pso(num_particles=row["particles"], max_iter=row["max_iter"], seed=row["seed"], ann=ann)
        
        ann =  ANN(layers=row["architecture"], Xdata=X_test.to_numpy(), Ydata=y_test.to_numpy())
        ann.fill_weights(position)
        ann.forward_pass()
        results["particle_pos"].append(position)
        results["particle_cost"].append(cost)
        results["cost_fn"].append(row["cost_fn"])
        results["accuracy"].append(ann.get_accuracy(output_activation=get_output_activation(row["architecture"])))
        results["seed"].append(row["seed"])
        results["session_name"].append(session_name)
        results["time"].append(get_time_str())
    
    write_experiments(db_path="experiments_db", results=results)