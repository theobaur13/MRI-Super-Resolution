import torch
import torch.nn as nn

class ResidualDenseBlock(nn.Module):
    def __init__(self, in_channels, growth_channels):
        super(ResidualDenseBlock, self).__init__()
        self.leaky_relu = nn.LeakyReLU(0.2, True)

        self.conv1 = nn.Conv2d(in_channels, growth_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels + (growth_channels * 1), growth_channels, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels + (growth_channels * 2), growth_channels, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels + (growth_channels * 3), growth_channels, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels + (growth_channels * 4), in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x1 = self.leaky_relu(self.conv1(x))
        x2 = self.leaky_relu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.leaky_relu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.leaky_relu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))

        return x + x5 * 0.2
    
class RRDB(nn.Module):
    def __init__(self, in_channels, growth_channels):
        super(RRDB, self).__init__()
        self.rdb1 = ResidualDenseBlock(in_channels, growth_channels)
        self.rdb2 = ResidualDenseBlock(in_channels, growth_channels)
        self.rdb3 = ResidualDenseBlock(in_channels, growth_channels)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return x + out * 0.2

class Generator(nn.Module):
    def __init__(
            self,
            in_channels = 1,
            channels = 64,
            growth_channels = 32,
            rrdb_count = 1,
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

        # Final conv layers
        self.conv_hr = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv_last = nn.Conv2d(channels, out_channels=1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        identity = self.conv1(x)
        trunk = self.trunk_conv(self.rrdb_trunk(identity))
        x = identity + trunk

        x = self.leaky_relu(self.conv_hr(x))
        x = self.conv_last(x)
        return x

class Discriminator(nn.Module):
    def __init__(self,
            in_channels = 1
        ):
        super(Discriminator, self).__init__()

        def conv_block(in_feat, out_feat, stride):
            return nn.Sequential(
                nn.Conv2d(in_feat, out_feat, kernel_size=3, stride=stride, padding=1),
                nn.BatchNorm2d(out_feat),
                nn.LeakyReLU(0.2, inplace=True)
            )

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

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
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(100, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x