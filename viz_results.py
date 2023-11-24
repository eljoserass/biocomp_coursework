import pandas as pd
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import sys


"""
Tool used to automatically try different types of visualizations on the experiments results
"""


df = pd.read_csv("./experiments_db.csv")

parser = argparse.ArgumentParser(
                    prog='Explore different Visualizations',
                    description='Passed the name of the choosed visualization, show it',
                    epilog='options: line-cost, line-accuracy, scatter, box, bar-iterations, bar-objective, pair, heatmap, violin, 3D')

parser.add_argument('-v', '--viz-name', choices=['line-cost', 'line-accuracy', 'scatter', 
                                                 'box', 'bar-iterations', 'bar-objective',
                                                 'pair', 'heatmap', 'violin', '3d'], required=True)

parser.add_argument('-s', '--session-name')  

def line_cost(df):
    plt.figure(figsize=(10, 6))
    plt.plot(df["iterations"], df["particle_cost"], marker='o', linestyle='-', color='r')

    plt.xlabel('Iterations')
    plt.ylabel('Particle Cost')
    plt.title('Evolution of Particle Cost over Iterations')

    plt.grid(True)
    plt.show()

def line_accuracy(df):
    plt.figure(figsize=(10, 6))
    plt.plot(df["iterations"], df["accuracy"], marker='o', linestyle='-', color='b')

    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.title('Evolution of Accuracy over Iterations')

    plt.grid(True)
    plt.show()

def scatter(df):
    plt.figure(figsize=(10, 6))
    plt.scatter(df["particle_cost"], df["particle_inertia"], c=df["accuracy"], cmap='viridis', s=50)

    plt.xlabel('Particle Cost')
    plt.ylabel('Particle Inertia')
    plt.title('Scatter Plot: Particle Cost vs Particle Inertia')

    plt.colorbar(label='Accuracy')
    plt.show()

def box(df):
    
    plt.figure(figsize=(10, 6))
    unique_architectures = df["architecture"].unique()
    architecture_mapping = {arch: i for i, arch in enumerate(unique_architectures)}

    for i in enumerate(unique_architectures):
        print(i)
    df_architecture_id_replaced = df.copy()
    df_architecture_id_replaced["architecture"] = df["architecture"].replace(architecture_mapping)
    sns.boxplot(x=df_architecture_id_replaced["architecture"], y=df["accuracy"])
    
    plt.xlabel('Architecture')
    plt.ylabel('Accuracy')
    plt.title('Box Plot: Accuracy across Different Architectures')

    plt.show()


def bar_iterations(df):
    plt.figure(figsize=(10, 6))
    plt.bar(df["iterations"], df["correct_predictions"], color='skyblue')

    plt.xlabel('Iterations')
    plt.ylabel('Correct Predictions')
    plt.title('Bar Chart: Correct Predictions at Different Iterations')

    plt.show()

def bar_objective(df):
    plt.figure(figsize=(10, 6))
    plt.bar(df["objective"], df["accuracy"], color='salmon')

    plt.xlabel('Objective')
    plt.ylabel('Accuracy')
    plt.title('Bar Chart: Objective vs Accuracy')

    plt.show()

def pair(df):
    sns.pairplot(df[['particle_cost', 'particle_inertia', 'c1', 'c2', 'accuracy']], height=2)
    plt.show()

def heatmap(df):
    plt.figure(figsize=(10, 8))
    df = df.drop(["cost_fn", "architecture", "correct_predictions", "objective", "session_name", "particle_pos", "time"], axis=1)
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')

    plt.title('Heatmap: Correlation Matrix')
    plt.show()

def violin(df):
    plt.figure(figsize=(10, 6))
    sns.violinplot(x=df["cost_fn"], y=df["accuracy"])

    plt.xlabel('Cost Function')
    plt.ylabel('Accuracy')
    plt.title('Violin Plot: Cost Function vs Accuracy')

    plt.show()

def three_dim(df):

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(df["particle_inertia"], df["particle_cost"], df["accuracy"], c='blue', marker='o')

    ax.set_xlabel('Particle Inertia')
    ax.set_ylabel('Particle Cost')
    ax.set_zlabel('Accuracy')
    ax.set_title('3D Scatter Plot: Particle Inertia, Particle Cost, and Accuracy')

    plt.show()


# Map functions of the plot to name of the argument
viz_dict = {
    'line-cost': line_cost,
    'line-accuracy': line_accuracy,
    'scatter': scatter,
    'box': box,
    'bar-iterations': bar_iterations,
    'bar-objective': bar_objective,
    'pair': pair,
    'heatmap': heatmap,
    'violin': violin,
    '3d': three_dim
    
}

args = parser.parse_args()


# Check if session name is passed
unique_session_name = df["session_name"].unique()
if "session_name" in args:
    if args.session_name in unique_session_name:
        df = df[df["session_name"] == args.session_name]
        print (args.session_name)

# Call Selected function by the user and pass dataframe
viz_dict[args.viz_name](df)