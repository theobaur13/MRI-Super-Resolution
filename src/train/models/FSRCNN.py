import torch.nn as nn

class FSRCNN(nn.Module):
    def __init__(self, in_channels=1, d=56, s=12, out_channels=1):
        super(FSRCNN, self).__init__()

        self.feature_extraction = nn.Sequential(
            nn.Conv2d(in_channels, d, kernel_size=5, stride=1, padding=2),
            nn.PReLU(d)
        )

        self.shrink = nn.Sequential(
            nn.Conv2d(d, s, kernel_size=1, stride=1, padding=0),
            nn.PReLU(s)
        )

        self.mapping = nn.Sequential(
            nn.Conv2d(s, s, kernel_size=3, stride=1, padding=1),
            nn.PReLU(s),
            nn.Conv2d(s, s, kernel_size=3, stride=1, padding=1),
            nn.PReLU(s),
            nn.Conv2d(s, s, kernel_size=3, stride=1, padding=1),
            nn.PReLU(s),
            nn.Conv2d(s, s, kernel_size=3, stride=1, padding=1),
            nn.PReLU(s)
        )

        self.expand = nn.Sequential(
            nn.Conv2d(s, d, kernel_size=1, stride=1, padding=0),
            nn.PReLU(d)
        )

        self.output = nn.Conv2d(d, out_channels, kernel_size=5, stride=1, padding=2)

    def forward(self, x):
        out = self.feature_extraction(x)
        out = self.shrink(out)
        out = self.mapping(out)
        out = self.expand(out)
        out = self.output(out)
        return out