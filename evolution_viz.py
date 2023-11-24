import plotly.graph_objects as go
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import os
import argparse

"""
Program with plotly and PCA to visualize the evolution of the 
particles per iterations in the space with the accuracy and standard deviation

Takes as parameter the experiment id
"""

parser = argparse.ArgumentParser(
                    prog='Experiment Evolution Visualization',
                    description='Open a Localhost interface to visualize particles evolution over iterations',
                    epilog='--experiment_id evolution_viz/<experiment_id>')

parser.add_argument('-e', '--experiment_id')  


args = parser.parse_args()

experiment_id = 0 if args.experiment_id == None else args.experiment_id

folder_path = f"./evolution_viz/{experiment_id}"

if not os.path.exists(folder_path):
    print ("No experiments made")
    exit (84)

dfs = []

# Load dataframes iterations

for filename in sorted(os.listdir(folder_path)):
    if filename.endswith(".csv"):
        file_path = os.path.join(folder_path, filename)
        df = pd.read_csv(file_path)
        dfs.append(df)

fig = go.Figure()

# Perform PCA on each particle, in all of the dataframes
# Create data object to put in the Scatter3D
for i, df in enumerate(dfs):

    positions = df['position']
    accuracy_values = df['accuracy']

    positions = positions.apply(lambda x: np.fromstring(x[1:-1], sep=', '))

    pca = PCA(n_components=3)
    reduced_positions = pca.fit_transform(positions.tolist())

    normalized_accuracy = (accuracy_values - accuracy_values.min()) / (accuracy_values.max() - accuracy_values.min())

    fig.add_trace(
        go.Scatter3d(
            visible=False,
            mode='markers',
            marker=dict(
                size=10,
                color=normalized_accuracy,
                colorscale='Viridis',
                opacity=0.8,
                colorbar=dict(title='Accuracy')
            ),
            name=f'Iteration {i}',
            x=reduced_positions[:, 0],
            y=reduced_positions[:, 1],
            z=reduced_positions[:, 2]
        )
    )

fig.data[0].visible = True

# Create stepper to move along iterations
steps = []
for i in range(len(fig.data)):
    step = dict(
        method="update",
        args=[{"visible": [False] * len(fig.data)},
              {"title": f"Standard Deviation {dfs[i]['standard_deviation'][0]} -- Iteration {i}"}],
    )
    step["args"][0]["visible"][i] = True
    steps.append(step)

sliders = [dict(
    active=0,
    currentvalue={"prefix": "Iteration: "},
    pad={"t": 50},
    steps=steps,
)]

fig.update_layout(
    sliders=sliders,
    scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z'
    )
)

fig.show()