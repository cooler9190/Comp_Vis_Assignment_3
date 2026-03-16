import torch
import json
from torch import nn

# Import dataloaders
from load_CIFAR10 import train_dataloader, validation_dataloader


def train_and_validate_CIFAR10(model, train_dataloader, validation_dataloader, loss_fn, optimizer, device, epochs=15):
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
            train_loss += loss.item() * X.size(0) # Accumulate loss
            correct_train += (outputs.argmax(1) == y).type(torch.float).sum().item() # Count correct predictions

        history['train_loss'].append(train_loss / total_train) # Average loss for the epoch
        history['train_accuracy'].append(correct_train / total_train) # Accuracy for the epoch

        # Validation phase
        model.eval() # Set model to evaluation mode
        val_loss = 0.0
        correct_val = 0
        total_val = len(validation_dataloader.dataset)

        with torch.no_grad(): # Disable gradient calculation for validation
            for X, y in validation_dataloader:
                X, y = X.to(device), y.to(device)

                outputs = model(X)
                loss = loss_fn(outputs, y)

                val_loss += loss.item() * X.size(0) # Accumulate validation loss
                correct_val += (outputs.argmax(1) == y).type(torch.float).sum().item() # Count correct predictions


        history['validation_loss'].append(val_loss / total_val) # Average validation loss for the epoch
        history['validation_accuracy'].append(correct_val / total_val) # Validation accuracy for the epoch

        print(f"Train - Loss: {history['train_loss'][-1]:>4f}, Accuracy: {(100*history['train_accuracy'][-1]):>0.1f}%")
        print(f"Validation - Loss: {history['validation_loss'][-1]:>4f}, Accuracy: {(100*history['validation_accuracy'][-1]):>0.1f}%\n")

    return history

def run_and_save_results(ModelToTrain, model_filename):
    # Get accelerator
    if hasattr(torch, 'accelerator') and torch.accelerator.is_available():
        device = torch.accelerator.current_accelerator().type
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Using {device} device")

    # Instantiate the model and move it to the appropriate device (GPU or CPU)
    model = ModelToTrain().to(device)
    print(f"Model architecture: {model}")
    print(f"Training model saving history to {model_filename}")

    # Define the loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Run the training and validation loop
    print("Starting training and validation...")
    history = train_and_validate_CIFAR10(model, train_dataloader, validation_dataloader, loss_fn, optimizer, device, epochs=15)

    # Save the training history to a JSON file
    with open(model_filename, 'w') as f:
        json.dump(history, f, indent=4)
    
    # Save trained model weights (optional)
    weight_filename = model_filename.replace(".json", ".pth")
    torch.save(model.state_dict(), weight_filename)
    
    print(f"Training complete. History saved to {model_filename}")
    print(f"Model weights saved to {weight_filename}")
