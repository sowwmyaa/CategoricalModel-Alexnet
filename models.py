import torch
import torch.nn as nn


# Every neural network in PyTorch inherits from nn.Module.
class AlexNet(nn.Module):

    # Our dataset has 8 emotion classes.
    def __init__(self, num_classes=7):
        super().__init__()      # Initialize the parent class

# ====================================================================
        # Feature Extractor
        # This part extracts features from the input image.
        self.features = nn.Sequential(
            # =====================================================================================
            # conv1 - Input : 227 x 227 x 3
            # Output: 55 x 55 x 96
            nn.Conv2d(
                in_channels=3,      # RGB image has 3 channels
                out_channels=96,    # Learn 96 different filters
                kernel_size=11,     # Each filter is 11x11
                stride=4,           # Move filter 4 pixels at a time
                padding=0           # No zero-padding
            ),

            # Adds non-linearity.
            # Without ReLU, multiple Conv layers behave like one linear layer.
            nn.ReLU(inplace=True),

            # Downsamples the feature map.
            # Keeps the strongest features while reducing computations.
            # Output: 27 x 27 x 96
            nn.MaxPool2d(
                kernel_size=3,
                stride=2
            ),
            # =====================================================================================
            # conv2 - Input : 27 x 27 x 96
            # Output: 27 x 27 x 256
            nn.Conv2d(
                in_channels=96,
                out_channels=256,   # Learn 256 higher level filters
                kernel_size=5,
                stride=1,
                padding=2           # Keeps spatial size unchanged
            ),

            nn.ReLU(inplace=True),

            # Output: 13 x 13 x 256
            nn.MaxPool2d(
                kernel_size=3,
                stride=2
            ),
            # =====================================================================================
            # conv3 -Input : 13 x 13 x 256
            # Output: 13 x 13 x 384
            nn.Conv2d(
                in_channels=256,
                out_channels=384,
                kernel_size=3,
                stride=1,
                padding=1           # Keeps output size same
            ),

            nn.ReLU(inplace=True),
            # =====================================================================================
            # conv4 -Input : 13 x 13 x 384
            # Output: 13 x 13 x 384
            nn.Conv2d(
                in_channels=384,
                out_channels=384,
                kernel_size=3,
                stride=1,
                padding=1
            ),

            nn.ReLU(inplace=True),
            # =====================================================================================
            # conv5 - Input : 13 x 13 x 384
            # Output: 13 x 13 x 256
            nn.Conv2d(
                in_channels=384,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1
            ),

            nn.ReLU(inplace=True),

            # Final downsampling.
            # Output: 6 x 6 x 256
            nn.MaxPool2d(
                kernel_size=3,
                stride=2
            )

        )

# ====================================================================

        # ---------------- Classifier ----------------
        # Takes the extracted features and predicts the emotion.
        self.classifier = nn.Sequential(

            # Prevents overfitting by randomly turning off neurons during training.
            nn.Dropout(p=0.5),

            # Input : 6 x 6 x 256 = 9216 features
            # Output: 4096 neurons
            nn.Linear(
                in_features=6 * 6 * 256,
                out_features=4096
            ),

            nn.ReLU(inplace=True),

            # Again helps reduce overfitting.
            nn.Dropout(p=0.5),

            # Input : 4096
            # Output: 4096
            nn.Linear(
                in_features=4096,
                out_features=4096
            ),

            nn.ReLU(inplace=True),

            # Final classification layer.
            # One output for each emotion class.
            nn.Linear(
                in_features=4096,
                out_features=num_classes
            )

        )

# ====================================================================
    def forward(self, x):  # x  is input image variables, check dataset.py

        # Pass image through convolutional layers.
        x = self.features(x)

        # Convert 6x6x256 feature maps into a single vector.
        x = torch.flatten(x, start_dim=1)

        # Pass flattened features through fully connected layers.
        x = self.classifier(x)

        return x
