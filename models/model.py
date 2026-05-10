# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SmallPDTCN(nn.Module):
    def __init__(self, num_frame_features, num_fft_features, hidden_channels=32):
        super(SmallPDTCN, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=num_frame_features, out_channels=hidden_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, padding=2, dilation=2)
        self.conv3 = nn.Conv1d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, padding=4, dilation=4)
        
        self.fft_fc = nn.Linear(num_fft_features, 16)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels + 16, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, temporal_x, fft_x):
        # temporal_x shape: (Batch, Features, Frames)
        x = F.relu(self.conv1(temporal_x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Global Average Pooling
        x = torch.mean(x, dim=2) 
        
        # FFT features
        f = F.relu(self.fft_fc(fft_x))
        
        # Combine and Classify
        combined = torch.cat((x, f), dim=1)
        out = self.classifier(combined)
        return out