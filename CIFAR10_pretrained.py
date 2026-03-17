import torch
from torch import nn

# We increase the number of kernels in the convolutional layers of the LeNet-5 architecture.
# This allows the model to learn more complex and abstract features from the input images, which can improve its performance on the CIFAR-10 dataset.

# NOTE: Because the number of output channels from conv2 increases from 16 to 32,
# the input features to the first fully connected layer (fc1) also need to be updated from 16*5*5 to 32*5*5 to match the new output dimensions of conv2.
class LeNet5VariantPretrained(nn.Module):
    def __init__(self):
        super().__init__()

        # Block 1: Input(3, 32, 32) -> Conv(16, 28, 28) -> Pool(16, 14, 14)
        # Using 5x5 kernels, no padding, max pooling better for image classification and object detection tasks
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 2: Input(16, 14, 14) -> Conv(32, 10, 10) -> Pool(32, 5, 5)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Flatten layer to convert 3D feature maps to 1D feature vector for fully connected layers
        self.flatten = nn.Flatten()

        # Fully connected layers
        # Adjusted input features to match the new output dimensions of conv2 (32 channels * 5 height * 5 width = 800 features)
        self.fc1 = nn.Linear(in_features=32*5*5, out_features=120)
        self.relu3 = nn.ReLU()

        # Dropout layer with a 45% probability
        dropout_probability = 0.45
        self.dropout1 = nn.Dropout(p=dropout_probability)

        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.relu4 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.45)

        # Output layer with 10 output features to match the number of classes in CIFAR-10 dataset
        self.fc3 = nn.Linear(in_features=84, out_features=10)

        # Apply custom weight initialization
        self._initialize_weights()

        # Define path to weights
        pretrained_weights_path = "history_variant_cifar100.pth"

        # Load the state dictionary from the file
        pretrained_dict = torch.load(pretrained_weights_path, weights_only=True)

        # Filter out the wights and biases of the output layer (fc3) since the number of output features in the pretrained model (20 for CIFAR-100) is different from the current model (10 for CIFAR-10)
        filtered_dict = {k: v for k, v in pretrained_dict.items() if not k.startswith("fc3")}

        # Load remaining wights
        self.load_state_dict(filtered_dict, strict=False)
        
    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)

        # Applying dropout after activation functions
        x = self.dropout1(self.relu3(self.fc1(x)))
        x = self.dropout2(self.relu4(self.fc2(x)))
        x = self.fc3(x)

        # Returning raw logits, as CrossEntropyLoss will apply softmax internally
        #x = nn.Softmax(dim=1)
        return x

    def _initialize_weights(self):
        # Iterate through all modules in the network
        for m in self.modules():
            # Apply Kaiming Uniform initialization to convolutional and linear layers
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                # 'nonlinearity' tells PyTorch to optimize the gain for ReLU
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                # Initialize biases to zero if they exist
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
