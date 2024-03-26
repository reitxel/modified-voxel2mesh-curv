import torch.nn as nn  
from IPython import embed 

class UNetLayer(nn.Module):
    def __init__(self, num_channels_in, num_channels_out, ndims, batch_norm=True, seblock=False):
        super(UNetLayer, self).__init__()
        conv_op = nn.Conv2d if ndims == 2 else nn.Conv3d
        batch_nrom_op = nn.BatchNorm2d if ndims == 2 else nn.BatchNorm3d
        
        self.conv1 = conv_op(num_channels_in, num_channels_out, kernel_size=3, padding=1)
        self.conv2 = conv_op(num_channels_out, num_channels_out, kernel_size=3, padding=1)
        
        # add dropout layer
        dropout_prob = 0.3
        self.dropout = nn.Dropout(dropout_prob)

        self.norm1 = batch_nrom_op(num_channels_out)
        self.norm2 = batch_nrom_op(num_channels_out)

        self.relu = nn.LeakyReLU(negative_slope=0.01)

        # Create a conv layer for matching dimensions if input and output channels are different
        if num_channels_in != num_channels_out:
            self.match_dimensions = conv_op(num_channels_in, num_channels_out, kernel_size=1)
        else:
            self.match_dimensions = None
            
        if seblock:
            self.se_block = SEBlock3D(num_channels_out)
        else:
            self.se_block = None

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        
        # adding the dropout before the last convolutional layer
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu(out)

        # Apply SE block if it exists
        if self.se_block is not None:
            out = self.se_block(out)

        if self.match_dimensions is not None:
            residual = self.match_dimensions(residual)
            
        out = out + residual
        return out


# Squeeze-and-Excitation Block
class SEBlock3D(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)

    
