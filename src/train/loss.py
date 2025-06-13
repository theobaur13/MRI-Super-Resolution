import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import vgg19, VGG19_Weights

class CompositeLoss(nn.Module):
    def __init__(self, weights=None):
        super().__init__()
        # Default weights (no pixel_alpha now since we're only using L1)
        self.weights = weights or {
            "edge": 0.7,
            "pixel": 0.3,
            "perceptual": 1.0,
        }
        self.l1 = nn.L1Loss()

        # Perceptual loss setup (VGG19 features)
        self.perceptual_vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features[:36].eval().to("cuda")
        for param in self.perceptual_vgg.parameters():
            param.requires_grad = False

        self.vgg_normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    def perceptual_loss(self, sr, hr):
        if sr.shape[1] == 1:
            sr = sr.repeat(1, 3, 1, 1)
            hr = hr.repeat(1, 3, 1, 1)

        sr_norm = torch.stack([self.vgg_normalize(s) for s in sr])
        hr_norm = torch.stack([self.vgg_normalize(h) for h in hr])

        sr_feat = self.perceptual_vgg(sr_norm)
        hr_feat = self.perceptual_vgg(hr_norm)

        return F.l1_loss(sr_feat, hr_feat)

    def sobel_edges(self, img):
        sobel_x = torch.tensor([[1, 0, -1],
                                [2, 0, -2],
                                [1, 0, -1]], dtype=torch.float32, device=img.device).view(1, 1, 3, 3)

        sobel_y = torch.tensor([[1, 2, 1],
                                [0, 0, 0],
                                [-1, -2, -1]], dtype=torch.float32, device=img.device).view(1, 1, 3, 3)

        gx = F.conv2d(img, sobel_x, padding=1)
        gy = F.conv2d(img, sobel_y, padding=1)
        grad = torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)
        return grad

    def edge_loss(self, sr, hr):
        edge_map = self.sobel_edges(hr)
        return torch.mean(edge_map * torch.abs(sr - hr))

    def pixel_loss(self, sr, hr):
        return self.l1(sr, hr)  # Only L1 loss used here

    def forward(self, sr, hr):
        loss = 0.0

        if self.weights["pixel"] > 0:
            loss += self.weights["pixel"] * self.pixel_loss(sr, hr)

        if self.weights["perceptual"] > 0:
            loss += self.weights["perceptual"] * self.perceptual_loss(sr, hr)

        if self.weights["edge"] > 0:
            loss += self.weights["edge"] * self.edge_loss(sr, hr)

        return loss

def gradient_penalty(discriminator, real, fake, device="cuda", lambda_gp=10):
    batch_size = real.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    interpolated = (alpha * real + (1 - alpha) * fake).requires_grad_(True)

    pred = discriminator(interpolated)

    gradients = torch.autograd.grad(
        outputs=pred,
        inputs=interpolated,
        grad_outputs=torch.ones_like(pred),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    gradients = gradients.view(batch_size, -1)
    gradients_norm = gradients.norm(2, dim=1)
    gp = lambda_gp * ((gradients_norm - 1) ** 2).mean()
    return gp