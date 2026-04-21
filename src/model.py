import torch
import torch.nn as nn

class GeneNet(nn.Module):
    def __init__(self, input_dim):
        super(GeneNet, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()  # 0(저항성) ~ 1(반응성) 확률
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        return self.classifier(x)
