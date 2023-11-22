import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch

# class ComparisonNN(nn.Module):
#     def __init__(self, input_size, hidden_size_tanh, hidden_size_relu, hidden_size_sigmoid, output_size):
#         super(ComparisonNN, self).__init__()

#         self.fc1_tanh = nn.Linear(input_size, hidden_size_tanh)
#         self.fc2_relu = nn.Linear(hidden_size_tanh, hidden_size_relu)
#         self.fc3_sigmoid = nn.Linear(hidden_size_relu, hidden_size_sigmoid)
#         self.fc4_softmax = nn.Linear(hidden_size_sigmoid, output_size)

#         self.tanh = nn.Tanh()
#         self.relu = nn.ReLU()
#         self.sigmoid = nn.Sigmoid()
#         # self.softmax = nn.Softmax(dim=1)

#     def forward(self, x):
#         x = self.tanh(self.fc1_tanh(x))
#         x = self.relu(self.fc2_relu(x))
#         x = self.sigmoid(self.fc3_sigmoid(x))
#         # x = self.softmax(self.fc4_softmax(x))
#         return x

# input_size = 4
# hidden_size_tanh = 8
# hidden_size_relu = 4
# hidden_size_sigmoid = 2
# output_size = 2
# learning_rate = 0.001
# batch_size = len(X_train)
# epochs = 5

# torch.manual_seed(4)

# X_train_np = np.array(X_train, dtype=np.float32)

# tensor_train = torch.tensor(X_train_np)

# labels = torch.tensor(y_train.values).long()

# dataset = TensorDataset(tensor_train, labels)
# dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=True)

# model = ComparisonNN(input_size=4, hidden_size_tanh=8, hidden_size_relu=4, hidden_size_sigmoid=2, output_size=2)
# criterion = nn.BCEWithLogitsLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.001)


# epochs = 5  # Adjust the number of epochs based on your specific problem
# for epoch in range(epochs):
#     model.train()

#     for features, labels in dataloader:
#         optimizer.zero_grad()
#         outputs = model(features)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

#     if (epoch + 1) % 100 == 0:
#         print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

# # Make predictions on the training data
# model.eval()
# with torch.no_grad():
#     predictions = model(X_test).argmax(dim=1)

# # Convert predictions and labels to numpy arrays for evaluation
# predictions = predictions.numpy()
# labels = labels.numpy()

# # Calculate accuracy
# accuracy = (predictions == y_test).mean()
# print(f"Training Accuracy: {accuracy * 100:.2f}%")
