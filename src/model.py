import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class GatedFusion(nn.Module):
    def __init__(self, c_rgb, c_geom):
        super().__init__()
        self.proj = nn.Conv2d(c_geom, c_rgb, 1) 
        self.gate = nn.Sequential(nn.Conv2d(c_rgb*2, 1, 1), nn.Sigmoid())
        
    def forward(self, rgb, geom):
        geom_proj = self.proj(geom)
        cat = torch.cat([rgb, geom_proj], dim=1)
        g = self.gate(cat)
        return (g * geom_proj) + ((1-g) * rgb)

class DeepDSPNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Stream A: RGB
        resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        self.enc1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool) 
        self.enc2 = resnet.layer1 # 64 ch
        self.enc3 = resnet.layer2 # 128 ch
        self.enc4 = resnet.layer3 # 256 ch
        
        # Stream B: Geometry (Edges + Depth)
        self.geo1 = nn.Conv2d(2, 64, 3, padding=1) 
        self.geo2 = nn.Conv2d(64, 128, 3, padding=1, stride=2)
        self.geo3 = nn.Conv2d(128, 256, 3, padding=1, stride=2)
        
        # Fusion
        self.fus2 = GatedFusion(64, 64)   
        self.fus3 = GatedFusion(128, 128) 
        self.fus4 = GatedFusion(256, 256) 
        
        # Decoder
        self.up4 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.up3 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.up2 = nn.ConvTranspose2d(64, 32, 2, 2)
        
        self.final = nn.Conv2d(32, 1, 3, padding=1)
        self.final_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, rgb, edge, depth):
        geom_input = torch.cat([edge, depth], dim=1)

        # RGB Stream
        e1 = self.enc1(rgb)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        
        # Geometry Stream
        g1 = F.interpolate(F.relu(self.geo1(geom_input)), size=e2.shape[2:])
        g2 = F.interpolate(F.relu(self.geo2(g1)), size=e3.shape[2:])
        g3 = F.interpolate(F.relu(self.geo3(g2)), size=e4.shape[2:])
        
        # Fusion
        f4 = self.fus4(e4, g3)
        f3 = self.fus3(e3, g2)
        f2 = self.fus2(e2, g1)
        
        # Decoder
        d = self.up4(f4) + f3
        d = self.up3(d) + f2
        d = self.up2(d)        
        
        out = self.final(d)
        return self.final_upsample(out)