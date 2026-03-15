import torch
from torch import nn
from CIFAR10_model1 import LeNet5Color
# Get accelerator device (GPU if available, otherwise CPU)
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

# Instantiate the model and move it to the appropriate device (GPU or CPU)
model = LeNet5Color().to(device)

# Define the loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print(model)