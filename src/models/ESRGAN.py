import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

class RDB():
    def __init__(self, in_channels, growth_channels):
        super().__init__()
        kernel = (3, 3)
        stride = (1, 1)
        padding = (1, 1)
        LRelu_slope = 0.2

        self.conv1 = nn.Conv2d(in_channels, growth_channels, kernel_size=kernel, stride=stride, padding=padding)
        self.conv2 = nn.Conv2d(in_channels + (growth_channels * 1), growth_channels, kernel_size=kernel, stride=stride, padding=padding)
        self.conv3 = nn.Conv2d(in_channels + (growth_channels * 2), growth_channels, kernel_size=kernel, stride=stride, padding=padding)
        self.conv4 = nn.Conv2d(in_channels + (growth_channels * 3), growth_channels, kernel_size=kernel, stride=stride, padding=padding)
        self.conv5 = nn.Conv2d(in_channels + (growth_channels * 4), in_channels, kernel_size=kernel, stride=stride, padding=padding)

        self.leaky_relu = nn.LeakyReLU(LRelu_slope, True)

    def forward(self, x):
        identity = x  

        x1 = self.leaky_relu(self.conv1(x))
        x2 = self.leaky_relu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.leaky_relu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.leaky_relu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.leaky_relu(self.conv5(torch.cat((x, x1, x2, x3, x4), 1)))

        return identity + x5
    
class RRDB():
    def __init__(self, in_channels, growth_channels, rrdb_count=3):
        super().__init__()
        self.rdbs = nn.Sequential(*[RDB(in_channels, growth_channels) for _ in range(rrdb_count)])

    def forward(self, x):
        identity = x
        x = self.rdbs(x)
        return identity + x

class Generator():
    def __init__(self,
            in_channels = 2,
            out_channels = 2,
            channels = 64,
            growth_channels = 32,
            rdb_count = 3,
            rrdb_count = 23,
        ):
        super().__init__()
        
        kernel = (3, 3)
        stride = (1, 1)
        padding = (1, 1)
        LRelu_slope = 0.2

        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=kernel, stride=stride, padding=padding)
        self.rrdbs = nn.Sequential(*[RRDB(channels, growth_channels, rdb_count) for _ in range(rrdb_count)])
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=kernel, stride=stride, padding=padding)

        self.upsample = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=kernel, stride=stride, padding=padding),
            nn.PixelShuffle(2),
            nn.LeakyReLU(LRelu_slope, inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=kernel, stride=stride, padding=padding),
            nn.LeakyReLU(LRelu_slope, inplace=True)
        )
        self.conv4 = nn.Conv2d(channels, out_channels, kernel_size=kernel, stride=stride, padding=padding)

    def forward(self, x):
        identity = self.conv1(x)
        x = self.rrdbs(identity)
        x = self.conv2(x) + identity

        x = self.upsample(x)
        x = self.conv3(x)
        x = self.conv4(x)

        return x

class Discriminator():
    def __init__(self,
            in_channels = 2,
            out_channels = 1,
            kernel_size = (3, 3),
            stride = (1, 1),
            padding = (1, 1),
            leaky_relu_slope = 0.2,
        ):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.LeakyReLU(leaky_relu_slope, inplace=True),

            nn.Conv2d(64, 64, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(leaky_relu_slope, inplace=True),

            nn.Conv2d(64, 64, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(leaky_relu_slope, inplace=True),

            nn.Conv2d(64, 128, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(leaky_relu_slope, inplace=True),

            nn.Conv2d(128, 128, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(leaky_relu_slope, inplace=True),

            nn.Conv2d(128, 256, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(leaky_relu_slope, inplace=True),

            nn.Conv2d(256, 256, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(leaky_relu_slope, inplace=True),

            nn.Conv2d(256, 512, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(leaky_relu_slope, inplace=True),

            nn.Conv2d(512, 512, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(leaky_relu_slope, inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.Linear(512 * 8 * 8, 100),
            nn.LeakyReLU(leaky_relu_slope, inplace=True),
            nn.Linear(100, out_channels),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class PerceptualLoss():
    pass

class ContentLoss():
    pass

class AdversarialLoss():
    pass