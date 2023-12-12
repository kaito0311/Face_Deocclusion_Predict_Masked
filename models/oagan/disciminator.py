import os 

import torch 

import torch
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    def __init__(self, input_size = 112, enable_face_component_loss = False):
        super(Discriminator, self).__init__()
        self.channel = 3
        self.input_size = input_size
        self.enable_face_component_loss = enable_face_component_loss
        # Images layer
        self.images_conv0 = nn.Conv2d(self.channel, 32, kernel_size=7, stride=2) # First conv must have large kernel size
        self.images_conv1 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.images_conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2)
        self.images_conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2)
        self.images_fc1 = nn.Linear(6400, 256)
        self.images_fc2 = nn.Linear(256, 1)

        if self.enable_face_component_loss:
            # Eyes layer
            self.eyes_conv0 = nn.Conv2d(self.channel, 16, kernel_size=7, stride=2) # First conv must have large kernel size
            self.eyes_conv1 = nn.Conv2d(16, 32, kernel_size=3, stride=2)
            self.eyes_conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
            self.eyes_conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2)
            self.eyes_fc1 = nn.Linear(3200, 128)
            self.eyes_fc2 = nn.Linear(128, 1)

            # Nose layer
            self.nose_conv0 = nn.Conv2d(self.channel, 16, kernel_size=7, stride=2) # First conv must have large kernel size
            self.nose_conv1 = nn.Conv2d(16, 32, kernel_size=3, stride=2)
            self.nose_conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
            self.nose_conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2)
            self.nose_fc1 = nn.Linear(3200, 128)
            self.nose_fc2 = nn.Linear(128, 1)

            # Mouth layer
            self.mouth_conv0 = nn.Conv2d(self.channel, 16, kernel_size=7, stride=2) # First conv must have large kernel size
            self.mouth_conv1 = nn.Conv2d(16, 32, kernel_size=3, stride=2)
            self.mouth_conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
            self.mouth_conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2)
            self.mouth_fc1 = nn.Linear(3200, 128)
            self.mouth_fc2 = nn.Linear(128, 1)

            # Face layer
            # self.face_conv0 = nn.Conv2d(self.channel, 16, kernel_size=7, stride=2)
            # self.face_conv1 = nn.Conv2d(16, 32, kernel_size=3, stride=2)
            # self.face_conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
            # self.face_conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2)
            # self.face_fc1 = nn.Linear(7 * 7 * 128, 128)
            # self.face_fc2 = nn.Linear(128, 1)

    def forward(self, images):
        # Input in range [-1, 1]
        bs, _, _, _ = images.size()

        

        # Images branch
        h0_0 = F.leaky_relu(self.images_conv0(images))
        h0_1 = F.leaky_relu(self.images_conv1(h0_0))
        h0_2 = F.leaky_relu(self.images_conv2(h0_1))
        h0_3 = F.leaky_relu(self.images_conv3(h0_2))
        h0_4 = h0_3.view(bs, -1)
        h0_5 = F.leaky_relu(self.images_fc1(h0_4))
        h0_6 = self.images_fc2(h0_5)

        if self.enable_face_component_loss:
            eyes = images[:, :, 32:57, 22:91]
            nose = images[:, :, 50:82, 40:69]
            mouth = images[:, :, 83:103, 38:75]
            # face = images[:, :, 17:83, 14:83]

            eyes = F.interpolate(eyes, size=(self.input_size, self.input_size), mode='bilinear')
            nose = F.interpolate(nose, size=(self.input_size, self.input_size), mode='bilinear')
            mouth = F.interpolate(mouth, size=(self.input_size, self.input_size), mode='bilinear')
            # face = F.interpolate(face, size=(self.input_size, self.input_size), mode='bilinear')

            # Eyes branch
            h1_0 = F.leaky_relu(self.eyes_conv0(eyes))
            h1_1 = F.leaky_relu(self.eyes_conv1(h1_0))
            h1_2 = F.leaky_relu(self.eyes_conv2(h1_1))
            h1_3 = F.leaky_relu(self.eyes_conv3(h1_2))
            h1_3 = h1_3.view(bs, -1)
            h1_4 = F.leaky_relu(self.eyes_fc1(h1_3))
            h1_5 = self.eyes_fc2(h1_4)

            # Nose branch
            h2_0 = F.leaky_relu(self.nose_conv0(nose))
            h2_1 = F.leaky_relu(self.nose_conv1(h2_0))
            h2_2 = F.leaky_relu(self.nose_conv2(h2_1))
            h2_3 = F.leaky_relu(self.nose_conv3(h2_2))
            h2_3 = h2_3.view(bs, -1)
            h2_4 = F.leaky_relu(self.nose_fc1(h2_3))
            h2_5 = self.nose_fc2(h2_4)

            # Mouth branch
            h3_0 = F.leaky_relu(self.mouth_conv0(mouth))
            h3_1 = F.leaky_relu(self.mouth_conv1(h3_0))
            h3_2 = F.leaky_relu(self.mouth_conv2(h3_1))
            h3_3 = F.leaky_relu(self.mouth_conv3(h3_2))
            h3_3 = h3_3.view(bs, -1)
            h3_4 = F.leaky_relu(self.mouth_fc1(h3_3))
            h3_5 = self.mouth_fc2(h3_4) * 2
            return h0_6, h1_5, h2_5, h3_5

        # Face branch
        # h4_0 = F.leaky_relu(self.face_conv0(face))
        # h4_1 = F.leaky_relu(self.face_conv1(h4_0))
        # h4_2 = F.leaky_relu(self.face_conv2(h4_1))
        # h4_3 = F.leaky_relu(self.face_conv3(h4_2))
        # h4_3 = h4_3.view(bs, -1)
        # h4_4 = F.leaky_relu(self.face_fc1(h4_3))
        # h4_5 = self.face_fc2(h4_4)

        return (h0_6,)#, h1_5, h2_5, h3_5