import plotly.graph_objects as go
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import os

# Path to the folder containing CSV files
folder_path = "evolution_viz/0"

# Initialize a list to store dataframes from each CSV file
dfs = []

# Iterate through CSV files in the folder
for filename in sorted(os.listdir(folder_path)):
    if filename.endswith(".csv"):
        file_path = os.path.join(folder_path, filename)
        df = pd.read_csv(file_path)
        dfs.append(df)

# Create figure
fig = go.Figure()

# Add traces, one for each slider step
# print (len(dfs))

for i, df in enumerate(dfs):
    # Extract the 'position' and 'accuracy' columns for dimensionality reduction
    positions = df['position']
    accuracy_values = df['accuracy']
    # Convert the string representation of lists to actual lists
    # print (df["iterations"][0])
    positions = positions.apply(lambda x: np.fromstring(x[1:-1], sep=', '))

    # Apply PCA for dimensionality reduction to 3D
    pca = PCA(n_components=3)
    reduced_positions = pca.fit_transform(positions.tolist())

    # Normalize the accuracy values to be in the range [0, 1]
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
            name=f'Time Step {i}',
            x=reduced_positions[:, 0],
            y=reduced_positions[:, 1],
            z=reduced_positions[:, 2]
        )
    )

# Make the first trace visible
fig.data[0].visible = True

# Create and add slider
steps = []
for i in range(len(fig.data)):
    step = dict(
        method="update",
        args=[{"visible": [False] * len(fig.data)},
              {"title": f"Time Step {i}"}],  # layout attribute
    )
    step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
    steps.append(step)

sliders = [dict(
    active=0,
    currentvalue={"prefix": "Time Step: "},
    pad={"t": 50},
    steps=steps,
)]

fig.update_layout(
    sliders=sliders,
    scene=dict(
        xaxis_title='Principal Component 1',
        yaxis_title='Principal Component 2',
        zaxis_title='Principal Component 3'
    )
)

fig.show()
