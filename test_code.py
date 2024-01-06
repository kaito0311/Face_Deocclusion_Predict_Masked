import os 

import cv2
import torch 
import numpy as np 
from PIL import Image 
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader

from config import cfg
from models.oagan.generator import OAGAN_Generator 
from dataset.dataloader import FaceRemovedMaskedDataset, FaceDataset
from models.pre_deocclusion.de_occlu_syn import FaceDeocclusionModel 


# transforms = T.Compose([
#             T.Resize((112,112)),
#             T.ToTensor(),
#             T.Normalize(mean=[0.5], std=[0.5])
#         ])

# model = OAGAN_Generator(
#     pretrained_encoder="/home1/data/tanminh/NML-Face/pretrained/r160_imintv4_statedict.pth",
#     pretrain_deocclu_model= "/home1/data/tanminh/Face_Deocclusion_Predict_Masked/pretrained/ckpt_gen_lastest.pt",
#     freeze_deocclu_model= True
# )
# model.load_state_dict(torch.load("all_experiments/pretrained_deocclu_training/second_experiment/ckpt/ckpt_gen_lastest.pt", map_location="cpu"))
# model.to("cpu")
# model.eval()
# while True:
#     path = input("input path: ")
#     try:
#         tensor = (Image.open(path))
#         tensor = transforms(tensor)
#         tensor = torch.unsqueeze(tensor, 0)

#         masked, restore_image, restore_image_wo_mask = model.predict(tensor.to("cpu"))
#         np.save("sub.npy", (restore_image_wo_mask - tensor).detach().numpy())
#         restore_image_wo_mask = restore_image_wo_mask.detach().cpu().numpy()
#         save_res_wo_mask = np.array(127.5 * (restore_image_wo_mask[0] + 1.0), dtype= np.uint8)
#         img = np.transpose(save_res_wo_mask, (1, 2, 0))
#         cv2.imwrite("out.jpg", img)
#         cv2.imwrite("in.jpg", cv2.imread(path))
        
#         print("hello")
#     except Exception as e: 
#         print(str(e))
# exit()


def process_torch(sub):
    sub = torch.abs(sub)
    sub /= 2.0 
    sub = sub[:, 0, :, :] * 0.2989 + sub[:, 1, :, :] *  0.5870 + sub[:, 2, :, :] * 0.1140

    return sub 



sub= process_torch(torch.from_numpy(np.load("sub.npy")))
print(sub.shape)
print(torch.unsqueeze(sub, 1).shape)
out = np.array(255 * sub[0].detach().numpy(), dtype= np.uint8)
cv2.imwrite("sub_torch.jpg", out)
exit()

transforms = T.Compose([
            T.Resize((112,112)),
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5])
        ])

model = OAGAN_Generator(
    pretrained_encoder="/home1/data/tanminh/NML-Face/pretrained/r160_imintv4_statedict.pth",
    pretrain_deocclu_model= "/home1/data/tanminh/Face_Deocclusion_Predict_Masked/pretrained/ckpt_gen_lastest.pt",
    freeze_deocclu_model= True
)
model.load_state_dict(torch.load("all_experiments/pretrained_deocclu_training/second_experiment/ckpt/ckpt_gen_lastest.pt", map_location="cpu"))
model.to("cpu")
model.eval()
tensor = (Image.open("images/val/masked/100009762790189_face_3.jpg"))
tensor = transforms(tensor)
tensor = torch.unsqueeze(tensor, 0)

masked, restore_image, restore_image_wo_mask = model.predict(tensor.to("cpu"))
restore_image_wo_mask = restore_image_wo_mask.detach().cpu().numpy()
save_res_wo_mask = np.array(127.5 * (restore_image_wo_mask[0] + 1.0), dtype= np.uint8)
img = np.transpose(save_res_wo_mask, (1, 2, 0))
cv2.imwrite("out.jpg", img)
np.save("sub.npy", (restore_image_wo_mask - tensor).detach().numpy())
print("hello")
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
