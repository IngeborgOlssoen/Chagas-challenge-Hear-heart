import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import torchvision.models as models
from torchvision.models import resnet50, ResNet50_Weights


# ----------------------------
# Audio Classification Model
# ----------------------------
class AudioClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        conv_layers = []
        self.bn0 = nn.BatchNorm2d(1)
        # First Convolution Block with Relu and Batch Norm. Use Kaiming Initialization
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2))
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(32)
        self.mp1 = nn.MaxPool2d(2)
        self.dp1 = nn.Dropout(p=0.15)
        init.kaiming_normal_(self.conv1.weight, a=0.1)
        self.conv1.bias.data.zero_()
        conv_layers += [self.conv1, self.bn1, self.relu1,  self.mp1]

        # Second Convolution Block
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(32)
        self.mp2 = nn.MaxPool2d(2)
        self.dp2 = nn.Dropout(p=0.1)
        init.kaiming_normal_(self.conv2.weight, a=0.1)
        self.conv2.bias.data.zero_()
        conv_layers += [self.conv2, self.bn2, self.relu2, self.mp2, self.dp2]

        # Third Convolution Block
        self.conv3 = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(32)
        self.dp3 = nn.Dropout(p=0.1)
        init.kaiming_normal_(self.conv3.weight, a=0.1)
        self.conv3.bias.data.zero_()
        conv_layers += [self.conv3, self.bn3, self.relu3, self.dp3]

        # Fourth Convolution Block
        self.conv4 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu4 = nn.ReLU()
        self.bn4 = nn.BatchNorm2d(64)
        init.kaiming_normal_(self.conv4.weight, a=0.1)
        self.conv4.bias.data.zero_()
        conv_layers += [self.conv4, self.bn4, self.relu4]

        # Linear Classifier
        self.ap = nn.AdaptiveAvgPool2d(output_size=1)
        
        # Wide features layer
        #self.lin_wide = nn.Linear(28, 20)  # Wide features layer (age, sex, etc.)

        # Final linear layer
        self.lin = nn.Linear(64, 1)  # Adjust for the concatenated features from conv layers and wide features

        # Wrap the Convolutional Blocks
        self.conv = nn.Sequential(*conv_layers)
        self.dp = nn.Dropout(p=0.3)

    def forward(self, x, x1=None):
        # Run the convolutional blocks
        x = self.conv(x)

        # Adaptive pool and flatten for input to linear layer
        x = self.ap(x)
        x_all = x.view(x.shape[0], -1)
        
        # Process wide features (x1), if provided, through its own linear layer
        #if x1 is not None:
        #    x1 = self.lin_wide(x1)  # Process wide features (e.g., age, sex, etc.)
#
        #    # Concatenate the wide features with the output of the convolutional layers
        #    x_all = torch.cat((x_all, x1), dim=1)

        # Final linear layer for classification
        x_all = self.lin(x_all)

        return x_all.view(-1)  # ✅ at the end of each model



# ----------------------------
# Audio Classification Model
# ----------------------------
class CRNN(nn.Module):
    def __init__(self, input_size=16, hidden_size=64, num_layers=1):
        super().__init__()
        conv_layers = []

        # First Convolution Block with Relu and Batch Norm
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(2, 2), padding=(2, 2))
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(16)
        init.kaiming_normal_(self.conv1.weight, a=0.1)
        self.conv1.bias.data.zero_()
        conv_layers += [self.conv1, self.relu1, self.bn1]

        # Additional convolutional layers can be added here...

        # Linear Classifier
        self.ap = nn.AdaptiveAvgPool2d(output_size=1)
        self.lin = nn.Linear(in_features=64, out_features=1)

        # LSTM input size should match the flattened output of the conv layers
        self.lstm_input_size = 16  # This should match the number of channels (features) after the convolution

        # LSTM Layer
        self.lstm1 = nn.LSTM(input_size=self.lstm_input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

        # Wrap the Convolutional Blocks
        self.conv = nn.Sequential(*conv_layers)
        self.dp = nn.Dropout(p=0.3)

    def forward(self, x, x1=None):
        # Run the convolutional blocks
        x = self.conv(x)

        # Debug: check the shape of x after convolution
        #print(x.shape)  # Should print [batch_size, channels, height, width]

        # Flatten the output of conv layers to pass it to the LSTM
        # Flatten the height and width into sequence length, and channels as input size
        x = x.view(x.size(0), 7 * 1501, 16)  # [batch_size, sequence_length, input_size]

        # Apply the LSTM layer
        x, (h_n, c_n) = self.lstm1(x)  # Apply LSTM

        # Now, use the last hidden state for the Linear layer
        # Take the last hidden state (last timestep) from LSTM output
        x = h_n[-1]  # h_n is of shape [num_layers, batch_size, hidden_size]

        # Check the shape of x before passing it to the Linear layer
        #print(x.shape)  # Should be [batch_size, hidden_size]

        # Linear layer
        x = self.lin(x)

        return x.view(-1)  # ✅ at the end of each model



# pretrained model
class ResNet(nn.Module):
    def __init__(self, pretrained=True, num_classes=1):  # Binary classification
        super().__init__()

        weights = ResNet50_Weights.DEFAULT if pretrained else None
        base_model = resnet50(weights=weights)
        
        # Modify the first convolutional layer to accept 1 input channel
        base_model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
        layers = list(base_model.children())[:-1]  # Remove the last fully connected layer
        self.encoder = nn.Sequential(*layers)
        self.bn1 = nn.BatchNorm1d(2048)

        # Wide feature layer (e.g., from age, sex, etc.)
        self.wide = nn.Linear(28, 20)  # Adjusting size to match ResNet output (2048)

        # Final output layer
        self.lin = nn.Linear(2048 + 20, num_classes)  # Concatenate and pass to a 1-class output layer for binary classification

    def forward(self, x, x1=None):
        #print(f"Initial shape of x: {x.shape}")
        
        # Ensure x is properly shaped for ResNet
        x = x.transpose(2, 3)  # Reorder dimensions for ResNet
        #print(f"Shape of x after transpose: {x.shape}")

        batch_size = x.size(0)
        
        # Process the input through ResNet encoder
        x_all = self.encoder(x).view(batch_size, -1)  # Shape: [batch_size, 2048]
        #print(f"Shape of x_all after encoder: {x_all.shape}")

        # If wide features (x1) are provided, ensure x1 is reshaped to the correct size
        if x1 is not None:
            #print(f"Initial shape of x1: {x1.shape}")
            # Make x1 match the batch size by repeating it
            x1 = x1.repeat(batch_size, 1)  # Repeat the tensor along the batch dimension
            #print(f"Shape of x1 after repeating: {x1.shape}")
            x1 = self.wide(x1)  # Process wide features (age, sex, etc.)
            #print(f"Shape of x1 after wide layer: {x1.shape}")

            # Debugging: print shapes before concatenation
            #print(f"x_all shape: {x_all.shape}")  # Shape of ResNet features (2048)
            #print(f"x1 shape: {x1.shape}")  # Shape of wide features (processed through self.wide)

            x_all = torch.cat((x_all, x1), dim=1)  # Concatenate along the feature axis
            #print(f"Shape of x_all after concatenation: {x_all.shape}")

        # Final classification layer
        x_all = self.lin(x_all)  # Shape should be [batch_size, num_classes]
        #print(f"Shape of x_all after final linear layer: {x_all.shape}")
        return x_all.view(-1)  # ✅ at the end of each model

