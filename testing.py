import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn

from sklearn.metrics import confusion_matrix
from load_CIFAR10 import test_dataloader
from CIFAR10_model2 import LeNet5Variant2 as Best10Model
from CIFAR10_pretrained import LeNet5VariantPretrained as PretrainedModel

# Get accelerator
if hasattr(torch, 'accelerator') and torch.accelerator.is_available():
    device = torch.accelerator.current_accelerator().type
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

def loading_path(model_to_test):
    if model_to_test == Best10Model: return "variant2_results_and_weights/45_percent_dropout_probability/history_variant2.pth"
    return "variant_pretrained_results_and_weights/45_percent_dropout_probability/history_variant_pretrained.pth"

def test_model(model_to_test):
    model = model_to_test() # Instantiate the model.
    path = loading_path(model_to_test) # Get params file path.
    model.load_state_dict(torch.load(path)) # Load params.
    model.to(device) # Move model to the appropriate device (GPU or CPU).
    model.eval() # Set model to evaluation mode.
    print(f"Model architecture: {model}")

    # Store predictions.
    all_predictions = []
    all_labels = []
    correct = 0
    total = len(test_dataloader.dataset)

    with torch.no_grad(): # Disable gradient calculation for validation
        for images, labels in test_dataloader: # For each testing batch:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            # Forward pass only.
            predictions = model(images) # Predict labels for all images in batch, in parallel.
            _, prediction_classes = torch.max(predictions, 1) # Obtain the predicted classes.

            # Add up count of correctly predicted labels.
            correct += (prediction_classes == labels).sum().item()

            # Update total arrays
            all_predictions.extend(prediction_classes.cpu().numpy()) # Ensure predictions are on cpu and converted to numpy array, before adding to all predictions.
            all_labels.extend(labels.cpu().numpy()) # Ensure labels are on cpu and converted to numpy array, before adding to all labels.

    # Accuracy printing
    print(f"correct: {correct}, total: {total}")
    accuracy = (correct / total) * 100 # Accuracy in percentage (%)
    print(f"Accuracy: {accuracy:0.1f}%")

    # Confusion Matrix
    confusion = confusion_matrix(all_labels, all_predictions)

    # Start matplotlib figure with labels and title.
    plt.figure(figsize=(10, 8))
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    # Replace the axis ticks with class labels
    class_labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    plt.xticks(ticks=np.arange(10)+0.5, labels=class_labels, rotation=45)  # predicted labels
    plt.yticks(ticks=np.arange(10)+0.5, labels=class_labels, rotation=0)   # true labels

    # Generate heatmap using seaborn
    seaborn.heatmap(confusion, annot=True, fmt="d", cmap="Blues")

    # Show the image.
    plt.show()


test_model(Best10Model)
test_model(PretrainedModel)
