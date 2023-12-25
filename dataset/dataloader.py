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
        self.list_path_occlusion_object = os.listdir(path_occlusion_object) if path_occlusion_object is not None else None 

        if str(list_name_data_non_occlusion).endswith(".npy"):
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
        else:
            if maximum is not None:
                self.list_img_occlu = shuffle(
                    os.listdir(list_name_data_occlusion))[:maximum]
                self.list_img_non_occlu = shuffle(
                    os.listdir(list_name_data_non_occlusion))[:maximum]
            else:
                self.list_img_occlu = shuffle(
                    os.listdir(list_name_data_occlusion))[:maximum]
                self.list_img_non_occlu = shuffle(
                    os.listdir(list_name_data_non_occlusion))[:maximum]


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
    
    def mask_random(self, image, occlusion_object=None, ratio_height=-1):
        if ratio_height is None: 
            ratio_height = np.clip(np.random.rand(), 0.3, 0.8)
        
        # if ratio_width is None:
        #     ratio_width = np.clip(np.random.rand(), 0.1, 0.5) 
        ratio_width = ratio_height
        
        image = np.copy(image)
        

        height, width = image.shape[0], image.shape[1]

        if ratio_height == -1: 
            occ_height, occ_width = image.shape[:2]
            occ_height = min(height, occ_height) 
            occ_width = min(width, occ_width)
        else:
            occ_height, occ_width = int(height * ratio_height), int(width * ratio_width)

        row_start = np.random.randint(0, height - int(height * ratio_height))
        row_end = min(row_start + occ_height, height)

        col_start = np.random.randint(0, width - int(height * ratio_width))
        col_end = min(col_start + occ_width, width)

        if occlusion_object is not None:
            
            occlusion_object = cv2.resize(
                occlusion_object, (occ_width, occ_height))
            occlu_image, mask = occlusion_object[:, :, :3], occlusion_object[:, :, 3:]
            occlu_image = occlu_image[:, :, ::-1]

            image[row_start:row_end, col_start:col_end, :] = occlu_image * \
                mask + image[row_start:row_end,
                                    col_start:col_end, :] * (1 - mask)
        else:
            occlusion_noise = np.random.rand(occ_height, occ_width, 3)
            occlusion_noise = np.array(occlusion_noise * 255, dtype=np.uint8)
            image[row_start:row_end, col_start:col_end, :] = occlusion_noise

        return image
    
    def augment_occlusion(self, image):

        # for 4 channels npy
        mask_image = np.load(os.path.join(self.path_occlusion_object, random.choice(self.list_path_occlusion_object)))
        
        # for image
        # mask_image = cv2.imread(os.path.join(self.path_occlusion_object, random.choice(self.list_path_occlusion_object)))
        # mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2RGB)
        
        image_augment = self.mask_random(image, mask_image, ratio_height= None)
        image_augment = Image.fromarray(image_augment)

        return image_augment

    def augment_gauss(self, image):
        
        image_augment = self.mask_random(image, occlusion_object=None, ratio_height= None)
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
