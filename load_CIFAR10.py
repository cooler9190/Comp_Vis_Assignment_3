# Tutorial used: https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html#preparing-your-data-for-training-with-dataloaders

from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

# Load training set from CIFAR10 dataset, download if not already downloaded, and transform to tensor
full_training_data = datasets.CIFAR10(
    root="CIFAR10",
    train=True,
    download=True,
    transform=ToTensor()
)

# Load test set from CIFAR10 dataset, download if not already downloaded, and transform to tensor
test_data = datasets.CIFAR10(
    root="CIFAR10",
    train=False,
    download=True,
    transform=ToTensor()
)

labels_map = {
    0: "airplane",
    1: "automobile",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck",
}

# We split the training data into a training set and a validation set to evaluate the model's performance during training. 
# The training set will be used to train the model, while the validation set will be used to evaluate the model's performance on unseen data and to tune hyperparameters.
# A 80-20 split is common practice, to ensure NN has enough data to learn from, and to have a sufficient validation set for evaluation.
train_size = 40000
validation_size = 10000

# Train and validation sets are created by randomly splitting the full training data into two subsets of the specified sizes.
training_data, validation_data = random_split(full_training_data, [train_size, validation_size])

# Create dataloaders for training, validation, and test sets. Dataloaders are used to load data in batches during training and evaluation.
# Validation and Test sets are not shuffled to ensure consistent evaluation, while the training set is shuffled to improve model generalization 
# by exposing it to different data orders during each epoch.
train_dataloader = DataLoader(training_data, batch_size=32, shuffle=True)
validation_dataloader = DataLoader(validation_data, batch_size=32, shuffle=False)
test_dataloader = DataLoader(test_data, batch_size=32, shuffle=False)


# Display a batch of training data

train_features, train_labels = next(iter(train_dataloader))

# Expecting a batch of 32 images, each with 3 color channels (RGB) and dimensions 32x32 pixels, and a batch of 32 labels corresponding to the images.
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")

# Display the first image in the batch and its corresponding label
# Pytorch tensors are in the format (channels, height, width), but matplotlib expects images in the format (height, width, channels)
# so we permute the dimensions of the image tensor to match the expected format for displaying with matplotlib.
img = train_features[0].permute(1, 2, 0)
label_idx = train_labels[0].item()

plt.imshow(img)
plt.title(f"Label: {labels_map[label_idx]}")
plt.axis("off")
plt.show()