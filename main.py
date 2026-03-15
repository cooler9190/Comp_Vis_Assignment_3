import torch
import json
from torch import nn

# Import dataloaders
from load_CIFAR10 import train_dataloader, validation_dataloader

# Import model architectures

# Baseline
from CIFAR10_lenet import LeNet5Color as ModelToTrain
model_filename = "history_baseline.json"

# Variant 1: Dropout
# from CIFAR10_model1 import LeNet5Variant1 as ModelToTrain
# model_filename = "history_variant1.json"

# Variant 2: Increased number of kernels
# from CIFAR10_model2 import LeNet5Variant2 as ModelToTrain
# model_filename = "history_variant2.json"



def train_and_validate(model, train_dataloader, validation_dataloader, loss_fn, optimizer, device, epochs=15):
    # Dictionary to store training and validation metrics for each epoch
    history = {
        'train_loss': [], 'train_accuracy': [],
        'validation_loss': [], 'validation_accuracy': []
    }
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}\n-------------------------------")
        
        # Training phase
        model.train()  # Set model to training mode
        train_loss = 0.0
        correct_train = 0
        total_train = len(train_dataloader.dataset)
        
        for X, y in train_dataloader:
            X, y = X.to(device), y.to(device)

            # Forward pass
            outputs = model(X)
            loss = loss_fn(outputs, y)

            # Backpropagation
            optimizer.zero_grad() # Clear previous gradients
            loss.backward() # Compute gradients
            optimizer.step() # Update weights

            # Track metrics











# # Get accelerator device (GPU if available, otherwise CPU)
# device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
# print(f"Using {device} device")

# # Instantiate the model and move it to the appropriate device (GPU or CPU)
# model = LeNet5Color().to(device)

# # Define the loss function and optimizer
# loss_fn = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# print(model)