import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.backbones.imintv5 import make_decoder_layer, make_upsample_layer, ConvBNBlock
from models.backbones.imintv5 import iresnet160

class IdentityModel(torch.nn.Module):
    def __init__(self, input_size = 112):
        super().__init__()
        self.input_size = input_size
        assert self.input_size == 112, "Current support only input_size = 112"
        self.model = iresnet160(False)

    # Generator Network
    def forward(self, x):
        # Get feature
        x, x1, x2, x3, x4 = self.model(x)
        return x, x1, x2, x3, x4


class Encoder(nn.Module):
    def __init__(self, pretrained = None):
        super(Encoder, self).__init__()
        self.model = IdentityModel()
        if pretrained is not None: 
            self.model.load_state_dict(torch.load(pretrained))
    
    def forward(self, x):
        x, x_56, x_28, x_14, x_7 = self.model(x)
        norm_x = x / torch.norm(x, p=2, dim=1, keepdim=True)
        return norm_x, x_56, x_28, x_14, x_7

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.layer1 = make_decoder_layer(inplanes=512, planes=256, n_blocks=8, activation = 'PReLU')
        self.layer2 = make_decoder_layer(inplanes=256, planes=128, n_blocks=8, activation = 'PReLU')
        self.layer3 = make_decoder_layer(inplanes=128, planes=64, n_blocks=8, activation = 'PReLU')
        self.layer4 = make_decoder_layer(inplanes=64, planes=32, n_blocks=8, activation = 'PReLU')
        self.layer5 = make_decoder_layer(inplanes=32, planes=16, n_blocks=8, activation = 'PReLU')
        self.last_conv = ConvBNBlock(inplanes = 16, planes = 3, activation = None)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.tanh = nn.Tanh()

    def forward(self, feat, x_56, x_28, x_14, x_7):
        x = self.layer1(x_7) # Bx128x7x7
        x = self.upsample(x) # Bx128x14x14
        # x = torch.cat((x, x_14), dim=1)
        x = self.layer2(x) # Bx64x14x14
        x = self.upsample(x)  # Bx64x28x28
        # x = torch.cat((x, x_28), dim=1)
        x = self.layer3(x) # Bx32x28x28
        x = self.upsample(x) # Bx32x56x56
        # x = torch.cat((x, x_56), dim=1)
        x = self.layer4(x) # Bx16x56x56
        x = self.upsample(x) # Bx16x112x112
        x = self.layer5(x) # Bx3x112x112
        x = self.last_conv(x)
        x = self.tanh(x)
        return x



class FaceDeocclusionModel(nn.Module):
    def __init__(self, pretrain = None, freeze_encoder = True)-> None:
        super(FaceDeocclusionModel, self).__init__()
        
        self.encoder = Encoder()
        self.decoder = Decoder() 
        self.freeze_encoder = freeze_encoder 
        
        if self.freeze_encoder: 
            self.encoder.eval() 
            for p in self.encoder.parameters():
                p.requires_grad = False 
    def forward(self, x):
        feat, x_56, x_28, x_14, x_7 = self.encoder(x)
        out = self.decoder(feat, x_56, x_28, x_14, x_7)
        return feat, out