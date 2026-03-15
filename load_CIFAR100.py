# Tutorial used: https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html#preparing-your-data-for-training-with-dataloaders

from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

# Load training set from CIFAR100 dataset, download if not already downloaded, and transform to tensor
full_training_data = datasets.CIFAR100(
    root="CIFAR100",
    train=True,
    download=True,
    transform=ToTensor()
)

# Load test set from CIFAR100 dataset, download if not already downloaded, and transform to tensor
test_data = datasets.CIFAR100(
    root="CIFAR100",
    train=False,
    download=True,
    transform=ToTensor()
)

# Extract the fine labels (specific classes) from the CIFAR-100 dataset, which contains 100 classes grouped into 20 superclasses.
# PyTorch stores them alphabetically in the .classes attribute
fine_labels_map = full_training_data.classes

# Group the final labels by their 20 coarse superclasses
coarse_mapping = {
    "aquatic mammals": ["beaver", "dolphin", "otter", "seal", "whale"],
    "fish": ["aquarium fish", "flatfish", "ray", "shark", "trout"],
    "flowers": ["orchids", "poppys", "roses", "sunflowers", "tulips"],
    "food containers": ["bottles", "bowls", "cans", "cups", "plates"],
    "fruit and vegetables": ["apples", "mushrooms", "oranges", "pears", "sweet peppers"],
    "household electrical devices": ["clock", "computer keyboard", "lamp", "telephone", "television"],
    "household furniture": ["bed", "chair", "couch", "table", "wardrobe"],
    "insects": ["bee", "beetle", "butterfly", "caterpillar", "cockroach"],
    "large carnivores": ["bear", "leopard", "lion", "tiger", "wolf"],
    "large man-made outdoor things": ["bridge", "castle", "house", "road", "skyscraper"],
    "large natural outdoor scenes": ["cloud", "forest", "mountain", "plain", "sea"],
    "large omnivores and herbivores": ["camel", "cattle", "chimpanzee", "elephant", "kangaroo"],
    "medium-sized mammals": ["fox", "porcupine", "possum", "raccoon", "skunk"],
    "non-insect invertebrates": ["crab", "lobster", "snail", "spider", "worm"],
    "people": ["baby", "boy", "girl", "man", "woman"],
    "reptiles": ["crocodile", "dinosaur", "lizard", "snake", "turtle"],
    "small mammals": ["hamster", "mouse", "rabbit", "shrew", "squirrel"],
    "trees": ["maple", "oak", "palm", "pine", "willow"],
    "vehicles 1": ["bicycle", "bus", "motorcycle", "pickup truck", "train"],
    "vehicles 2": ["lawn-mower", "rocket", "streetcar", "tank", "tractor"]
}

# Create a reverse lookup dictionary: fine label to coarse label
fine_to_coarse = {}
for coarse_label, fine_labels_list in coarse_mapping.items():
    for fine_label in fine_labels_list:
        fine_to_coarse[fine_label] = coarse_label

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

# Look up the fine and coarse label names
fine_label_name = fine_labels_map[label_idx]
coarse_label_name = fine_to_coarse[fine_label_name]

plt.imshow(img)
plt.title(f"Fine: {fine_label_name.title()} | Coarse: {coarse_label_name.title()}")
plt.axis("off")
plt.show()