import torch
import torch.nn as nn

class ResidualDenseBlock(nn.Module):
    def __init__(self, in_channels, growth_channels):
        super(ResidualDenseBlock, self).__init__()
        kernel = 3
        stride = 1
        padding = 1

        self.leaky_relu = nn.LeakyReLU(0.2, True)

        self.conv1 = nn.Conv2d(in_channels, growth_channels, kernel_size=kernel, stride=stride, padding=padding)
        self.conv2 = nn.Conv2d(in_channels + (growth_channels * 1), growth_channels, kernel_size=kernel, stride=stride, padding=padding)
        self.conv3 = nn.Conv2d(in_channels + (growth_channels * 2), growth_channels, kernel_size=kernel, stride=stride, padding=padding)
        self.conv4 = nn.Conv2d(in_channels + (growth_channels * 3), growth_channels, kernel_size=kernel, stride=stride, padding=padding)
        self.conv5 = nn.Conv2d(in_channels + (growth_channels * 4), in_channels, kernel_size=kernel, stride=stride, padding=padding)

    def forward(self, x):
        x1 = self.leaky_relu(self.conv1(x))
        x2 = self.leaky_relu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.leaky_relu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.leaky_relu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))

        return x + x5 * 0.2
    
class RRDB(nn.Module):
    def __init__(self, in_channels, growth_channels, rrdb_count=3):
        super(RRDB, self).__init__()
        self.blocks = nn.Sequential(*[
            ResidualDenseBlock(in_channels, growth_channels) for _ in range(rrdb_count)
        ])

    def forward(self, x):
        return x + self.blocks(x) * 0.2

class Generator(nn.Module):
    def __init__(
            self,
            in_channels = 2,
            out_channels = 2,
            channels = 64,
            growth_channels = 32,
            rdb_count = 3,
            rrdb_count = 23,
        ):
        super(Generator, self).__init__()
        
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=3, stride=1, padding=1)

        # RRDB blocks
        self.rrdb_trunk = nn.Sequential(*[
            RRDB(channels, growth_channels) for _ in range(rrdb_count)
        ])

        # After RRDB blocks
        self.trunk_conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)

        # Upsampling layers
        self.upsample = nn.Sequential(
            nn.Conv2d(channels, channels * 4, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(channels, channels * 4, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Final conv layers
        self.conv_hr = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv_last = nn.Conv2d(channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        identity = self.conv1(x)
        trunk = self.trunk_conv(self.rrdb_trunk(identity))
        x = identity + trunk

        x = self.upsample(x)
        x = self.leaky_relu(self.conv_hr(x))
        x = self.conv_last(x)
        return x

class Discriminator(nn.Module):
    def __init__(self,
            in_channels = 2,
            leaky_relu_slope = 0.2,
        ):
        super(Discriminator, self).__init__()

        def conv_block(in_feat, out_feat, stride):
            return nn.Sequential(
                nn.Conv2d(in_feat, out_feat, kernel_size=3, stride=stride, padding=1),
                nn.BatchNorm2d(out_feat),
                nn.LeakyReLU(leaky_relu_slope, inplace=True)
            )

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(leaky_relu_slope, inplace=True),

            conv_block(64, 64, stride=2),
            conv_block(64, 128, stride=1),
            conv_block(128, 128, stride=2),
            conv_block(128, 256, stride=1),
            conv_block(256, 256, stride=2),
            conv_block(256, 512, stride=1),
            conv_block(512, 512, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),  
            nn.Flatten(),
            nn.Linear(512 * 4 * 4, 100),
            nn.LeakyReLU(leaky_relu_slope, inplace=True),
            nn.Linear(100, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x