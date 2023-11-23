import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.widgets import Slider

# Path to the folder containing CSV files
folder_path = "evolution_viz/0"

# Initialize a list to store dataframes from each CSV file
dfs = []

# Iterate through CSV files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):
        file_path = os.path.join(folder_path, filename)
        df = pd.read_csv(file_path)
        dfs.append(df)

# Combine all dataframes into a single dataframe
data = pd.concat(dfs, ignore_index=True)

# Extract the 'position' and 'accuracy' columns for dimensionality reduction
positions = data['position']
accuracy_values = data['accuracy']

# Convert the string representation of lists to actual lists
positions = positions.apply(lambda x: np.fromstring(x[1:-1], sep=', '))

# Apply PCA for dimensionality reduction to 3D

pca = PCA(n_components=3)
reduced_positions = pca.fit_transform(positions.tolist())

# Normalize the accuracy values to be in the range [0, 1]
normalized_accuracy = (accuracy_values - accuracy_values.min()) / (accuracy_values.max() - accuracy_values.min())

# Create a 3D scatter plot with a color gradient
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(
    reduced_positions[:, 0],
    reduced_positions[:, 1],
    reduced_positions[:, 2],
    c=normalized_accuracy,
    cmap=cm.viridis,
    marker='o'
)

# Add a colorbar to show the mapping from values to colors
cbar = plt.colorbar(scatter)
cbar.set_label('Accuracy')

# Add labels and title
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
ax.set_title('3D Visualization with Color Gradient Based on Accuracy')

# Define the slider axes
ax_slider = plt.axes([0.1, 0.01, 0.65, 0.03], facecolor='lightgoldenrodyellow')
slider = Slider(ax_slider, 'Time Step', 0, len(dfs) - 1, valinit=0, valstep=1)

# Update function for the slider
def update(val):
    global scatter  # Declare scatter as a global variable
    
    timestep = int(slider.val)
    current_df = dfs[timestep]
    
    # Clear the existing scatter plot
    scatter.remove()
    
    # Update the scatter plot
    positions = current_df['position'].apply(lambda x: np.fromstring(x[1:-1], sep=', '))
    reduced_positions = pca.transform(positions.tolist())
    normalized_accuracy = (current_df['accuracy'] - current_df['accuracy'].min()) / (current_df['accuracy'].max() - current_df['accuracy'].min())
    
    scatter = ax.scatter(
        reduced_positions[:, 0],
        reduced_positions[:, 1],
        reduced_positions[:, 2],
        c=normalized_accuracy,
        cmap=cm.viridis,
        marker='o'
    )

    # Add a colorbar to show the mapping from values to colors
    cbar.update_normal(scatter)

# Attach the update function to the slider
slider.on_changed(update)

plt.show()

# Attach the update function to the slider
slider.on_changed(update)

plt.show()
