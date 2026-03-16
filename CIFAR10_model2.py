from torch import nn

# We increase the number of kernels in the convolutional layers of the LeNet-5 architecture.
# This allows the model to learn more complex and abstract features from the input images, which can improve its performance on the CIFAR-10 dataset.

# NOTE: Because the number of output channels from conv2 increases from 16 to 32, 
# the input features to the first fully connected layer (fc1) also need to be updated from 16*5*5 to 32*5*5 to match the new output dimensions of conv2.
class LeNet5Variant2(nn.Module):
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
        
        # First attempt: Adding Droput layer with a 50% probability - good balance between regularization and model capacity, leading to improved performance on the validation set compared to the baseline.
        # Second attempt: Adding Droput layer with a 25% probability - too low as it led to the model overfitting.
        # Third attempt: Adding Droput layer with a 40% probability - 2nd best result so far with peak validation acc of 65.9%(Epoch 13) and lowest validation loss of 0.99(Epoch 11), with a moderate gap between training and validation in Epoch 15
        # Fourth attempt: Adding Dropout layer with a 60% probability - validation acc sharply drops from Epoch 13 to Epoch 15
        # Fifth attempt: Adding Dropout layer with a 55% probability - slightly worse than 50%
        # Sixth attempt Adding Dropout layer with a 45% probability - best result overall, outperforming 40% with better results at Epoch 15 and much less gap between training and validation, indicating better generalization to unseen data.
        self.dropout1 = nn.Dropout(p=0.45)

        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.relu4 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.45)

        self.fc3 = nn.Linear(in_features=84, out_features=10)

        # Apply custom weight initialization
        self._initialize_weights()
    
    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        
        # Applying dropout after activation functions
        x = self.dropout1(self.relu3(self.fc1(x)))
        x = self.dropout2(self.relu4(self.fc2(x)))
        x = self.fc3(x)

        # Returning raw logits, as CrossEntropyLoss will apply softmax internally
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
