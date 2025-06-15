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
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2),
            nn.Dropout(0.1),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.Dropout(0.1),
        )
        self.ap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(32, 1)
        )

    def forward(self, x, x1=None):
        x = self.conv(x)
        x = self.ap(x)
        x = x.view(x.size(0), -1)
        return self.fc(x).view(-1)




# ----------------------------
# CRNN MODULE
# ----------------------------
class CRNN(nn.Module):
    def __init__(self, hidden_size=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.MaxPool2d(2)
        )
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


        # Run dummy input to infer the shape
        dummy_input = torch.zeros(1, 1, 12, 3000)  # [B, C=1, H=12, W=3000]
        with torch.no_grad():
            out = self.conv(dummy_input)
        _, C, H, _ = out.shape
        lstm_input_dim = C * H

        self.lstm = nn.LSTM(input_size=lstm_input_dim, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x, x1=None):
        x = self.conv(x)                        # [B, C, H, W]
        x = x.permute(0, 3, 1, 2).contiguous()  # [B, W, C, H]
        x = x.view(x.size(0), x.size(1), -1)    # [B, W, C×H]
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        return self.fc(x).view(-1)





# pretrained model
class ResNet(nn.Module):
    def __init__(self, pretrained=True, num_classes=1):
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        base_model = resnet50(weights=weights)

        # Adjust first layer for 1-channel ECG
        base_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Freeze all layers
        for param in base_model.parameters():
            param.requires_grad = False

        self.encoder = nn.Sequential(*list(base_model.children())[:-1])  # Remove final FC
        self.bn = nn.BatchNorm1d(2048)
        self.dropout = nn.Dropout(p=0.3)
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x, x1=None):
        x = x.transpose(2, 3)  # Convert [B, 1, 12, L] to [B, 1, L, 12]
        x = self.encoder(x).view(x.size(0), -1)
        x = self.bn(x)
        x = self.dropout(x)
        return self.fc(x).view(-1)


