# https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
import torch
import json
from torch import nn

# The model is deteriorating if no (significant) improvement is found,
# compared to the best (lowest) validation loss at that point.
def should_deteriorate(validation_loss, lowest_validation_loss, until_convergence):
    if validation_loss < lowest_validation_loss:
        if until_convergence: # See if relative improvement is significant.
            # We compare the current loss to the lowest record loss, instead of comparing to the (general) previous loss.
            # This is done as we want to measure the relative **improvement**, not the relative change.
            relative_improvement = (lowest_validation_loss - validation_loss) / lowest_validation_loss
            minimal_improvement = 1e-3
            if relative_improvement < minimal_improvement: return True # => No significant improvement.
        return False # => Model improved (significantly)
    return True # => Model didn't improve (significantly)

def train_and_validate(model, train_dataloader, validation_dataloader, loss_fn, optimizer, device, weight_filename, max_epochs=50, until_convergence=False):
    # Dictionary to store training and validation metrics for each epoch, including the deteriorated epoch metrics.
    # Therefore, the 'best' epoch count is 'length - patience'.
    history = {
        'train_loss': [], 'train_accuracy': [],
        'validation_loss': [], 'validation_accuracy': []
    }

    # Early Stopping condition: stop when validation loss hasn't improved for 'patient' epochs.
    patience = 5
    deterioration_counter = 0
    lowest_validation_loss = float('inf')

    # Loop for max_epochs long
    for epoch in range(max_epochs):
        print(f"Epoch {epoch+1}/{max_epochs}\n-------------------------------")
        
        # Training phase
        model.train()  # Set model to training mode
        train_loss = 0.0
        train_accuracy = 0
        train_length = len(train_dataloader.dataset)
        
        for images, labels in train_dataloader: # For each testing batch:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            # Forward pass
            predictions = model(images) # Predict labels for all images in batch, in parallel.
            loss = loss_fn(predictions, labels)

            # Backpropagation
            loss.backward() # Compute gradients
            optimizer.step() # Update weights
            optimizer.zero_grad() # Clear previous gradients

            # Track metrics
            train_loss += loss.item() * images.size(0) # Accumulate loss
            train_accuracy += (predictions.argmax(1) == labels).type(torch.float).sum().item() # Count only correct predictions (top-1)

        history['train_loss'].append(train_loss / train_length) # Average loss for the epoch
        history['train_accuracy'].append(train_accuracy / train_length) # Accuracy for the epoch

        # Validation phase
        model.eval() # Set model to evaluation mode
        validation_loss = 0.0
        validation_accuracy = 0
        validation_length = len(validation_dataloader.dataset)

        with torch.no_grad(): # Disable gradient calculation for validation
            for images, labels in validation_dataloader: # For each testing batch:
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

                # Forward pass
                predictions = model(images) # Predict labels for all images in batch, in parallel.
                loss = loss_fn(predictions, labels)

                # Track metrics
                validation_loss += loss.item() * images.size(0) # Accumulate validation loss
                validation_accuracy += (predictions.argmax(1) == labels).type(torch.float).sum().item() # Count only correct predictions (top-1)

        history['validation_loss'].append(validation_loss / validation_length) # Average validation loss for the epoch
        history['validation_accuracy'].append(validation_accuracy / validation_length) # Validation accuracy for the epoch

        # Printing training and validation data.
        print(f"Train - Loss: {history['train_loss'][-1]:>4f}, Accuracy: {(100*history['train_accuracy'][-1]):>0.1f}%")
        print(f"Validation - Loss: {history['validation_loss'][-1]:>4f}, Accuracy: {(100*history['validation_accuracy'][-1]):>0.1f}%\n")

        # Check for validation loss deterioration (=> early stop / until convergence).
        deteriorate = should_deteriorate(validation_loss, lowest_validation_loss, until_convergence)
        if deteriorate: deterioration_counter += 1 # Model getting worse (or plateauing).
        else: # Model improving (significantly).
            lowest_validation_loss = validation_loss # New lower (better) validation loss.
            deterioration_counter = 0 # Reset deterioration.
            torch.save(model.state_dict(), weight_filename) # Override saved trained model weights.

        # Early Stopping (or convergence), when model hasn't improved for 'patience' long.
        if deterioration_counter >= patience:
            print(f"Early stopping, saved epoch is: {epoch - deterioration_counter + 1}")
            break

    return history

def run_and_save_results(model_to_train, model_filename, train_dataloader, validation_dataloader, converge_mode=False, learning_rate=0.001):
    # Get accelerator
    if hasattr(torch, 'accelerator') and torch.accelerator.is_available():
        device = torch.accelerator.current_accelerator().type
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Using {device} device")

    # Instantiate the model and move it to the appropriate device (GPU or CPU)
    model = model_to_train().to(device)
    print(f"Model architecture: {model}")
    print(f"Training model saving history to {model_filename}")

    # Define the loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Run the training and validation loop
    print("Starting training and validation...")
    weight_filename = model_filename.replace(".json", ".pth")
    history = train_and_validate(model, train_dataloader, validation_dataloader, loss_fn, optimizer, device, weight_filename, max_epochs=50, until_convergence=converge_mode)

    # Save the training history to a JSON file
    with open(model_filename, 'w') as f:
        json.dump(history, f, indent=4)
    
    print(f"Training complete. History saved to {model_filename}")
    print(f"Model weights saved to {weight_filename}")
