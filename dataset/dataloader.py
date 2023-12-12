import os

import cv2
import torch
import torchvision
import numpy as np
from torch.utils import data
from PIL import Image


class FaceRemovedMaskedDataset(data.Dataset):
    def __init__(self, list_name_data_occlusion, list_name_data_non_occlusion, is_train=True, path_occlusion_object=None, augment_occlusion=False, augment_gauss=False) -> None:
        super().__init__()

        self.list_name_data_occlusion = list_name_data_occlusion
        self.list_name_data_non_occlusion = list_name_data_non_occlusion
        self.is_train = is_train
        self.path_occlusion_object = path_occlusion_object
        self.is_augment_occlusion = augment_occlusion
        self.is_augment_gauss = augment_gauss

    def augment_occlusion(self, image):
        raise NotImplemented("Co lam thi moi co an :))")

    def augment_gauss(self, image):
        raise NotImplemented("Co lam thi moi co an :))")

    def __getitem__(self, index):
        raise NotImplemented("Co lam thi moi co an :))")

    def __len__(self):
        raise NotImplemented("Co lam thi moi co an :))")

    pass

