
import pandas as pd
from sklearn.model_selection import train_test_split
from experiment_suite import run_experiments
import numpy as np
import os


"""
Executable that reads the files iniside experiments_config, and run every exeperiment of every config file in the folder
"""



# Load Data Parse columns, shuffle the rows and do a 80/20 split
df_banknote = pd.read_csv("data_banknote_authentication.csv",  header=None)
X = df_banknote.iloc[:, [0,1,2,3]]
Y = df_banknote.iloc[:, [4]]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42) 


# Get Files list
all_files = os.listdir("./experiments_config")

# Get the name of the experiment
csv_files_without_extension = [os.path.splitext(file)[0] for file in all_files if file.endswith(".csv")]

# Run experiment for each configuration file
for i in range(len(all_files)):
    print (f"config : {all_files[i]}")
    run_experiments(X_train, X_test, y_train, y_test, config_path=f"./experiments_config/{all_files[i]}", session_name=csv_files_without_extension[i], db_path="experiments_db.csv", evolution_viz=False)

# Run specifc experiment to write in evolution_viz folder for different type of visualization
run_experiments(X_train, X_test, y_train, y_test, config_path=f"./experiments_config/viz_evolution.csv", session_name="evolution", db_path="experiments_db.csv", evolution_viz=True)