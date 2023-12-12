import os
import random

import cv2
import torch
import torchvision
import numpy as np
from PIL import Image
from torch.utils import data
from sklearn.utils import shuffle
from torchvision import transforms as T

from config import cfg


class FaceRemovedMaskedDataset(data.Dataset):
    def __init__(self, list_name_data_occlusion, list_name_data_non_occlusion, root_dir,
                 is_train=True, path_occlusion_object=None, augment_occlusion=False, augment_gauss=False, maximum=None) -> None:
        super().__init__()

        self.list_name_data_occlusion = list_name_data_occlusion
        self.list_name_data_non_occlusion = list_name_data_non_occlusion
        self.is_train = is_train
        self.path_occlusion_object = path_occlusion_object
        self.is_augment_occlusion = augment_occlusion
        self.is_augment_gauss = augment_gauss
        self.root_dir = root_dir

        if maximum is not None:
            self.list_img_occlu = shuffle(
                np.load(list_name_data_occlusion))[:maximum]
            self.list_img_non_occlu = shuffle(
                np.load(list_name_data_non_occlusion))[:maximum]
        else:
            self.list_img_occlu = shuffle(
                np.load(list_name_data_occlusion))[:maximum]
            self.list_img_non_occlu = shuffle(
                np.load(list_name_data_non_occlusion))[:maximum]

        self.list_img_occlu = [os.path.join(
            root_dir, path) for path in self.list_img_occlu]
        self.list_img_non_occlu = [os.path.join(
            root_dir, path) for path in self.list_img_non_occlu]

        self.total_occlu = len(self.list_img_occlu)
        self.total_non_occlu = len(self.list_img_non_occlu)

        if self.is_train == "train":
            self.transforms = T.Compose([
                T.Resize((cfg.size_image, cfg.size_image)),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean=[0.5], std=[0.5])
            ])
        else:
            self.transforms = T.Compose([
                T.Resize((cfg.size_image, cfg.size_image)),
                T.ToTensor(),
                T.Normalize(mean=[0.5], std=[0.5])
            ])

    def augment_occlusion(self, image):
        image_augment = np.copy(np.array(image))
        index_start_row = np.random.choice(range(0, 100))
        distance = np.random.choice(range(50, 70))
        index_end_row = min(index_start_row + distance, 112)

        mask_image = cv2.imread(os.path.join(self.path_occlusion_object, random.choice(
            os.listdir(self.path_occlusion_object))))
        mask_image = cv2.resize(
            mask_image, (cfg.size_image, index_end_row - index_start_row))
        sum_mask_image = np.sum(mask_image, axis=2)
        sum_mask_image_1 = np.where(sum_mask_image == 0, 0, 1)
        sum_mask_image_1 = np.repeat(
            np.expand_dims(sum_mask_image_1, 2), 3, axis=2)
        sum_mask_image_0 = np.where(sum_mask_image == 0, 1, 0)
        sum_mask_image_0 = np.repeat(
            np.expand_dims(sum_mask_image_0, 2), 3, axis=2)

        image_augment[index_start_row:index_end_row, :, :] = mask_image * sum_mask_image_1 + \
            image_augment[index_start_row:index_end_row,
                          :, :] * sum_mask_image_0

        image_augment = Image.fromarray(image_augment)

        return image_augment

    def augment_gauss(self, image):
        image_augment = np.copy(np.array(image))
        index_start_row = np.random.choice(range(0, 100))
        distance = np.random.choice(range(50, 70))
        index_end_row = min(index_start_row + distance, 112)
        noise_value = np.array(
            np.random.normal(
                size=(index_end_row - index_start_row, cfg.size_image, 3)) * 255, dtype=np.uint8
        )
        image_augment[index_start_row:index_end_row, :, :] += noise_value
        image_augment = Image.fromarray(image_augment)
        return image_augment

    def __getitem__(self, index):
        path_occlu = self.list_img_occlu[int(index % self.total_occlu)]
        path_non_occlu = self.list_img_non_occlu[int(
            index % self.total_non_occlu)]

        occlu = Image.open(path_occlu)

        # add noise to image
        if np.random.rand() < cfg.noised_mask_ratio_occlu:
            occlu_augment = self.augment_gauss(occlu)
        elif np.random.rand() < cfg.synthetic_mask_ratio_occlu:
            occlu_augment = self.augment_occlusion(occlu)
        else:
            occlu_augment = occlu.copy()

        non_occlu = Image.open(path_non_occlu)
        # add noise to image
        if np.random.rand() < cfg.noised_mask_ratio_non_occlu:
            non_occlu_augment = self.augment_gauss(non_occlu)
        elif np.random.rand() < cfg.synthetic_mask_ratio_non_occlu:
            non_occlu_augment = self.augment_occlusion(non_occlu)
        else:
            non_occlu_augment = non_occlu.copy()

        occlu = self.transforms(occlu)
        occlu_augment = self.transforms(occlu_augment)

        non_occlu = self.transforms(non_occlu)
        non_occlu_augment = self.transforms(non_occlu_augment)

        return occlu, occlu_augment, non_occlu, non_occlu_augment

    def __len__(self):
        return max(self.total_non_occlu, self.total_occlu)
