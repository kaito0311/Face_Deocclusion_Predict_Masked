import os 

import torch 
import numpy as np 
from torch.utils.data import Dataset, DataLoader

from config import cfg
from models.oagan.generator import OAGAN_Generator 
from dataset.dataloader import FaceRemovedMaskedDataset, FaceDataset

from models.pre_deocclusion.de_occlu_syn import FaceDeocclusionModel 


model= FaceDeocclusionModel()

weight = (torch.load("pretrained/ckpt_gen_lastest.pt", map_location="cpu"))

model.load_state_dict(weight)
# print(list(model.state_dict().keys())[:100])
# print(list(weight.keys())[:100])


exit()



dataset = FaceDataset(
    path_list_name_data=cfg.valid_data_non_occlu,
    root_dir=cfg.ROOT_DIR,
    is_train= True,
    ratio_occlu= 1.0,
    path_occlusion_object="images/occlusion_object/clean_segment"
)


train_loader = DataLoader(
    dataset, batch_size=16, shuffle=True,
    num_workers=cfg.num_workers, drop_last=True)

train_iter = iter(train_loader) 

count = 0 
while True: 
    batch = next(train_iter)
    mask, occ, img = batch 

    print(mask.shape)
    print(occ.shape) 
    print(img.shape)

    print(torch.sum(img - occ))

    exit(0)
    count += 16

exit()

trainset = FaceRemovedMaskedDataset(
    list_name_data_occlusion=cfg.valid_data_non_occlu,  # NOTE
    list_name_data_non_occlusion=cfg.valid_data_non_occlu,
    root_dir=cfg.ROOT_DIR,
    is_train=True,
    path_occlusion_object="images/occlusion_object/clean_segment",
)

train_loader = DataLoader(
    trainset, batch_size=16, shuffle=True,
    num_workers=cfg.num_workers, drop_last=True)

train_iter = iter(train_loader) 

count = 0 
while True: 
    batch = next(train_iter)
    print(batch[0].shape, count)
    count += 16


exit() 

file_npy = np.load('/home1/data/tanminh/NML-Face/list_name_file/list_name_val_masked.npy')
root_dir = "/home1/data/FFHQ/StyleGAN_data256_jpg" 

val_dir = "/home1/data/tanminh/Face_Deocclusion_Predict_Masked/images/val/masked"
for name in file_npy: 
    print(name)
    cmd = f"cp {os.path.join(root_dir, name)} {val_dir}"
    os.system(cmd)

# gen = OAGAN_Generator(
#     pretrained_encoder= "/home1/data/tanminh/NML-Face/pretrained/r160_imintv4_statedict.pth",
#     arch_encoder= "r160",
# )
# gen.to("cuda")
# dummy_input = torch.rand(5, 3, 112, 112).to('cuda')

# feat, output = gen(dummy_input)
# print(output.shape)
