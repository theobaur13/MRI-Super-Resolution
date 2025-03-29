import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

class SRCNN_MRI(nn.Module):
    def __init__(self, num_channels=1):
        super(SRCNN_MRI, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=2)
        self.upsample = nn.Upsample(scale_factor=2.5, mode='bilinear', align_corners=False)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        x = self.upsample(x)
        return x
    
def training_loop(model, optimizer, criterion, epochs, LR, HR, output_dir=None):
    for epoch in tqdm(range(epochs)):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        output = model(LR)
        
        # Compute loss
        loss = criterion(output, HR)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

            if output_dir:
                torch.save(model.state_dict(), f"{output_dir}/model_epoch_{epoch + 1}.pth")