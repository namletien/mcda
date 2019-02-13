"""LeNet model for ADDA."""

import torch.nn.functional as F
from torch import nn

n_classes = 10
mid_dim = 500

class LeNetEncoder(nn.Module):
    """LeNet encoder model for ADDA."""

    def __init__(self):
        """Init LeNet encoder."""
        super(LeNetEncoder, self).__init__()

        self.restored = False

        self.encoder = nn.Sequential(
            nn.Linear(4096,1024),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1024,mid_dim),
        )

    def forward(self, input):
        """Forward the LeNet."""
        feat = self.encoder(input)
        return feat


class LeNetClassifier(nn.Module):
    """LeNet classifier model for ADDA."""

    def __init__(self):
        """Init LeNet encoder."""
        super(LeNetClassifier, self).__init__()
        self.fc2 = nn.Linear(mid_dim, n_classes)

    def forward(self, feat):
        """Forward the LeNet classifier."""
        out = F.dropout(F.relu(feat), training=self.training)
        out = self.fc2(out)
        return out
