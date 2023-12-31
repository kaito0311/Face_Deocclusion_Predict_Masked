import os

import torch
import torch.nn as nn
import numpy as np 
from torchvision import transforms as T

from models.backbones.imintv5 import make_decoder_layer, ConvBNBlock
from models.backbones.imintv5_custom import iresnet160_custom


class ResidualBlock(nn.Module):

    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True))

    def forward(self, x):
        return x + self.main(x)


class PredictMasked(torch.nn.Module):
    def __init__(self, conv_dim=64, repeat_num=6, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=7,
                                stride=1, padding=3, bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True))
        layers.append(nn.ReLU(inplace=True))
        curr_dim = conv_dim
        for i in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2,
                                    kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        for i in range(2):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim //
                                             2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim//2, affine=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        self.main = nn.Sequential(*layers)

        layers = []
        layers.append(nn.Conv2d(curr_dim, 1, kernel_size=7,
                                stride=1, padding=3, bias=False))
        # layers.append(nn.Sigmoid())
        self.attetion_reg = nn.Sequential(*layers)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        features = self.main(x)
        mask_map = self.attetion_reg(features)
        mask_map = self.sig(mask_map)
        return mask_map

    pass


class Encoder(torch.nn.Module):
    def __init__(self, pretrained=None, arch="r160", *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        if arch.lower() == "r160":
            self.model = iresnet160_custom(pretrained=False)

            if pretrained is not None:
                print("[INFO] Load weight: ", pretrained)
                self.model.load_state_dict(torch.load(pretrained))

    def forward(self, x):
        x, x_56, x_28, x_14, x_7 = self.model(x)
        norm_x = x / torch.norm(x, p=2, dim=1, keepdim=True)
        return norm_x, x_56, x_28, x_14, x_7


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.layer1 = make_decoder_layer(
            inplanes=512, planes=256, n_blocks=8, activation='PReLU')
        self.layer2 = make_decoder_layer(
            inplanes=256, planes=128, n_blocks=8, activation='PReLU')
        self.layer3 = make_decoder_layer(
            inplanes=128, planes=64, n_blocks=8, activation='PReLU')
        self.layer4 = make_decoder_layer(
            inplanes=64, planes=32, n_blocks=8, activation='PReLU')
        self.layer5 = make_decoder_layer(
            inplanes=32, planes=16, n_blocks=8, activation='PReLU')
        self.last_conv = ConvBNBlock(inplanes=16, planes=3, activation=None)
        self.upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=False)
        self.tanh = nn.Tanh()

    def forward(self, feat, x_56, x_28, x_14, x_7):
        x = self.layer1(x_7)  # Bx128x7x7
        x = self.upsample(x)  # Bx128x14x14
        # x = torch.cat((x, x_14), dim=1)
        x = self.layer2(x)  # Bx64x14x14
        x = self.upsample(x)  # Bx64x28x28
        # x = torch.cat((x, x_28), dim=1)
        x = self.layer3(x)  # Bx32x28x28
        x = self.upsample(x)  # Bx32x56x56
        # x = torch.cat((x, x_56), dim=1)
        x = self.layer4(x)  # Bx16x56x56
        x = self.upsample(x)  # Bx16x112x112
        x = self.layer5(x)  # Bx3x112x112
        x = self.last_conv(x)
        x = self.tanh(x)
        return x


class DeocclusionFaceGenerator(torch.nn.Module):
    def __init__(self, pretrained_encoder=None, arch_encoder=None, freeze_encoder=True, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.encoder = Encoder(
            pretrained=pretrained_encoder, arch=arch_encoder)
        self.preprocess_encoder = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=3,  # input 4 channel instead of 3 channels
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64, eps=1e-05),
            nn.PReLU(64)
        )
        self.decoder = Decoder()

        self.freeze_encoder = freeze_encoder

        if freeze_encoder:
            self.encoder.eval()
            for p in self.encoder.parameters():
                p.requires_grad = False

    def forward(self, x, masked):
        rich_feature_image = self.preprocess_encoder(
            torch.concat([x * masked, masked], dim=1))
        feat, x_56, x_28, x_14, x_7 = self.encoder.forward(rich_feature_image)
        out = self.decoder(feat, x_56, x_28, x_14, x_7)
        return feat, out


class OAGAN_Generator(torch.nn.Module):
    def __init__(self, pretrained_encoder, arch_encoder, freeze_encoder=True, image_size = 112, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        from model_sam import Model
        from config import cfg_sam
        self.predict_masked_model = Model(cfg_sam)
        self.predict_masked_model.setup()
        self.deocclusion_model = DeocclusionFaceGenerator(
            pretrained_encoder=pretrained_encoder, arch_encoder=arch_encoder, freeze_encoder=freeze_encoder)
        self.embed_text = torch.from_numpy(np.load("pretrained/feature_text.npy"))
        self.embed_text = self.embed_text.to("cuda" if torch.cuda.is_available() else "cpu")
        self.resize = T.Compose([T.Resize((image_size, image_size))])


    def forward(self, sam_image, unet_image):
        masked, _ = self.predict_masked_model(sam_image, None, self.embed_text)
        list_stack = [masked[i][0:1, :, :].unsqueeze(0) for i in range(len(masked))]
        masked = torch.vstack(list_stack)
        masked = self.resize(masked)
        feature, restore_image = self.deocclusion_model(unet_image, masked)
        restore_image = restore_image * masked + unet_image * (1.0 - masked)
        return feature, restore_image

    @torch.no_grad()
    def predict(self, sam_image, unet_image): 
        masked, _ = self.predict_masked_model(sam_image, None, self.embed_text)
        list_stack = [masked[i][0:1, :, :].unsqueeze(0) for i in range(len(masked))]
        masked = torch.vstack(list_stack)
        masked = self.resize(masked)
        feature, restore_image = self.deocclusion_model(unet_image, masked)
        restore_image = restore_image * masked + unet_image * (1.0 - masked)
        return masked, restore_image