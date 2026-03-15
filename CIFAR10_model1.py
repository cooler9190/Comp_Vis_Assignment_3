from torch import nn

# We introduce Dropout layers into the fully connected layers of the LeNet-5 architecture 
# to help prevent overfitting. Dropout randomly sets a fraction of the input units to zero during training, 
# which helps the model generalize better by preventing it from relying too heavily on any single feature or combination of features. 
class LeNet5Variant1(nn.Module):
    def __init__(self):
        super().__init__()

        # Block 1: Input(3, 32, 32) -> Conv(6, 28, 28) -> Pool(6, 14, 14)
        # Using 5x5 kernels, no padding, max pooling better for image classification and object detection tasks
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 2: Input(6, 14, 14) -> Conv(16, 10, 10) -> Pool(16, 5, 5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Flatten layer to convert 3D feature maps to 1D feature vector for fully connected layers
        self.flatten = nn.Flatten()

        # Fully connected layers
        # Input features are 16 channels * 5 height * 5 width = 400 features
        self.fc1 = nn.Linear(in_features=16*5*5, out_features=120)
        self.relu3 = nn.ReLU()

        # Adding Droput layer with a 50% probability
        self.dropout1 = nn.Dropout(p=0.5)

        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.relu4 = nn.ReLU()

        # Second Dropout layer
        self.dropout2 = nn.Dropout(p=0.5)

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
