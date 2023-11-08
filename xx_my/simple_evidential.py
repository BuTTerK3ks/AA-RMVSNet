import torch
import torch.nn as nn


class EvidentialNetwork(nn.Module):
	def __init__(self, input_dim, hidden_dim, num_classes):
		super(EvidentialNetwork, self).__init__()
		self.fc1 = nn.Linear(input_dim, hidden_dim)
		self.fc2 = nn.Linear(hidden_dim, num_classes)
		self.fc3 = nn.Linear(hidden_dim, num_classes)

	def forward(self, x):
		x = torch.relu(self.fc1(x))
		alpha = torch.exp(self.fc2(x))
		beta = torch.exp(self.fc3(x))
		return alpha, beta


# Define the model
input_dim = 2  # Example: 2 input features
hidden_dim = 64  # Example: 64 hidden units
num_classes = 2  # Example: Binary classification

model = EvidentialNetwork(input_dim, hidden_dim, num_classes)


# Define loss function (Negative Log-Likelihood for inverse gamma distribution)
def inverse_gamma_nll(alpha, beta, y):
	return alpha.log() - (alpha + 1) * y.log() - beta / y


# Define optimizer (you can choose any optimizer you like)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train_data = torch.rand(3,2)
target = torch.tensor(0)
num_epochs = 10

# Training loop (assuming you have some training data loaded as `train_data`)
for epoch in range(num_epochs):
	optimizer.zero_grad()
	alpha, beta = model(train_data)

	# Assuming `target` contains the true labels (0 or 1)
	loss = inverse_gamma_nll(alpha, beta, target.float())
	loss = torch.mean(loss)
	loss.backward()
	optimizer.step()

# After training, you can use the model to make predictions and estimate uncertainty
