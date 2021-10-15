import torch.nn as nn
import torch.nn.functional as F

class AtariWorldModel(nn.Module):
    """
    The general Idea:

    We begin by forming a Neural Network that takes as Input the four last frames of the
    game as well as the action that lead to this state. The network will be trained to 
    attempt to produce the frame that resulted from the action.

    """
    def __init__(self):
        super(AtariWorldModel, self).__init__()

        self.conv_dropout = nn.Dropout(0.15)
        self.skip = nn.Identity()

        # (_, 64, 105, 80) -> (_, 128, 53, 40)
        self.layer_norm1 = nn.LayerNorm([64, 105, 80])
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1)

        # (_, 128, 53, 40) -> (_, 256, 27, 20)        
        self.layer_norm2 = nn.LayerNorm([128, 53, 40])
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1)

        # (_, 256, 27, 20) -> (_, 256, 14, 10)
        self.layer_norm3 = nn.LayerNorm([256, 27, 20])
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=2)

        # (_, 256, 14, 10) -> (_, 256, 7, 5)
        self.layer_norm4 = nn.LayerNorm([256, 14, 10])
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=2)
        
        # (_, 256, 7, 5)   -> (_, 256, 4, 3)
        self.layer_norm5 = nn.LayerNorm([256, 7, 5])
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=2)
        
        # (_, 256, 4, 3)   -> (_, 256, 2, 2)
        self.layer_norm6 = nn.LayerNorm([256, 4, 3])
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=2)

    def forward(self, frames, action):

        # Convolutional Section of WorldModel
        x = F.relu(self.layer_norm1(self.conv1(self.conv_dropout(frames))))
        x = F.relu(self.layer_norm2(self.conv2(self.conv_dropout(x))))
        x = F.relu(self.layer_norm3(self.conv3(self.conv_dropout(x))))
        x = F.relu(self.layer_norm4(self.conv4(self.conv_dropout(x))))
        x = F.relu(self.layer_norm5(self.conv5(self.conv_dropout(x))))
        x = F.relu(self.layer_norm6(self.conv6(self.conv_dropout(x))))

        # Deconvolutional Section of WorldModel

        return x