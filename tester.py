
import pandas as pd
from sklearn.model_selection import train_test_split
from experiment_suite import run_experiments
import numpy as np
import os


df_banknote = pd.read_csv("data_banknote_authentication.csv",  header=None)

X = df_banknote.iloc[:, [0,1,2,3]]
Y = df_banknote.iloc[:, [4]]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

"""
Can do this in a loop, changing session name for giving it a title of what the experiments of taht csv are related,
and iterating through different .csv configs
"""


all_files = os.listdir("./experiments_config")

csv_files_without_extension = [os.path.splitext(file)[0] for file in all_files if file.endswith(".csv")]

for i in range(len(all_files)):
    print (f"config : {all_files[i]}")
    run_experiments(X_train, X_test, y_train, y_test, config_path=f"./experiments_config/{all_files[i]}", session_name=csv_files_without_extension[i], db_path="experiments_db.csv", evolution_viz=False)

run_experiments(X_train, X_test, y_train, y_test, config_path=f"./experiments_config/viz_evolution.csv", session_name="evolution", db_path="experiments_db.csv", evolution_viz=True)