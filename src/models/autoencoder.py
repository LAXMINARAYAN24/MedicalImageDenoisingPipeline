import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Convolutional block: Conv2d -> ReLU -> BatchNorm"""

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              padding=kernel_size // 2)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.bn(x)
        return x


class Encoder(nn.Module):
    """Encoder: Downsampling pathway"""

    def __init__(self, in_channels=1, initial_filters=32):
        super(Encoder, self).__init__()
        self.block1 = nn.Sequential(
            ConvBlock(in_channels, initial_filters),
            ConvBlock(initial_filters, initial_filters)
        )
        self.pool1 = nn.MaxPool2d(2, 2)

        self.block2 = nn.Sequential(
            ConvBlock(initial_filters, initial_filters * 2),
            ConvBlock(initial_filters * 2, initial_filters * 2)
        )
        self.pool2 = nn.MaxPool2d(2, 2)

        self.block3 = nn.Sequential(
            ConvBlock(initial_filters * 2, initial_filters * 4),
            ConvBlock(initial_filters * 4, initial_filters * 4)
        )
        self.pool3 = nn.MaxPool2d(2, 2)

    def forward(self, x):
        # First block
        x1 = self.block1(x)
        x = self.pool1(x1)

        # Second block
        x2 = self.block2(x)
        x = self.pool2(x2)

        # Third block
        x3 = self.block3(x)
        x = self.pool3(x3)

        return x, [x1, x2, x3]


class Decoder(nn.Module):
    """Decoder: Upsampling pathway with skip connections"""

    def __init__(self, out_channels=1, initial_filters=32):
        super(Decoder, self).__init__()

        # upconv outputs same ch as skip so concat = 2x; block reduces back to skip ch
        self.upconv3 = nn.ConvTranspose2d(initial_filters * 4, initial_filters * 4,
                                          kernel_size=2, stride=2)
        self.block3 = nn.Sequential(
            ConvBlock(initial_filters * 8, initial_filters * 2),
            ConvBlock(initial_filters * 2, initial_filters * 2)
        )

        self.upconv2 = nn.ConvTranspose2d(initial_filters * 2, initial_filters * 2,
                                          kernel_size=2, stride=2)
        self.block2 = nn.Sequential(
            ConvBlock(initial_filters * 4, initial_filters),
            ConvBlock(initial_filters, initial_filters)
        )

        self.upconv1 = nn.ConvTranspose2d(initial_filters, initial_filters,
                                          kernel_size=2, stride=2)
        self.block1 = nn.Sequential(
            ConvBlock(initial_filters * 2, initial_filters),
            ConvBlock(initial_filters, initial_filters)
        )

        self.final_conv = nn.Conv2d(initial_filters, out_channels, 1)

    def forward(self, x, skip_connections):
        x1, x2, x3 = skip_connections

        # Decoder block 3
        x = self.upconv3(x)
        x = torch.cat([x, x3], dim=1)
        x = self.block3(x)

        # Decoder block 2
        x = self.upconv2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.block2(x)

        # Decoder block 1
        x = self.upconv1(x)
        x = torch.cat([x, x1], dim=1)
        x = self.block1(x)

        # Final output
        x = self.final_conv(x)
        x = torch.tanh(x)  # Output in [-1, 1]

        return x


class AutoEncoder(nn.Module):
    """Complete AutoEncoder for medical image denoising"""

    def __init__(self, in_channels=1, initial_filters=32):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(in_channels, initial_filters)
        self.decoder = Decoder(in_channels, initial_filters)

    def forward(self, x):
        bottleneck, skip_connections = self.encoder(x)
        output = self.decoder(bottleneck, skip_connections)
        return output


class DenoisingAutoEncoder(nn.Module):
    """Denoising AutoEncoder with skip connections (U-Net style)"""

    def __init__(self, in_channels=1, initial_filters=32):
        super(DenoisingAutoEncoder, self).__init__()
        self.autoencoder = AutoEncoder(in_channels, initial_filters)

    def forward(self, x):
        return self.autoencoder(x)
