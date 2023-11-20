
import pandas as pd
from sklearn.model_selection import train_test_split
from experiment_suite import run_experiments

df_banknote = pd.read_csv("data_banknote_authentication.csv",  header=None)

X = df_banknote.iloc[:, [0,1,2,3]]
Y = df_banknote.iloc[:, [4]]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

run_experiments(X_train, X_test, y_train, y_test, config_path="config1.csv", session_name="test")