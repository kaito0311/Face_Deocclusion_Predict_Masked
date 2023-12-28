import os 

import cv2
import torch 
import numpy as np 
from torch.utils.data import Dataset, DataLoader

from config import cfg, cfg_sam
from models.oagan.generator import OAGAN_Generator 
from dataset.dataloader import FaceRemovedMaskedDataset, FaceDataset


from model_sam import Model 


sam = Model(cfg_sam)
sam.setup() 
sam.to("cuda")
sam.eval() 

embed_text = torch.from_numpy(np.load("pretrained/feature_text.npy"))
embed_text = embed_text.to("cuda")


image = cv2.imread("images/val/masked/100000166523059_face_3.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (1024, 1024))
input_tensor = torch.from_numpy(np.array(image/ 255.0, np.float32) )
input_tensor = torch.permute(input_tensor, (2, 0, 1))
input_tensor = torch.unsqueeze(input_tensor, 0)
print(input_tensor.shape)
input_tensor = input_tensor.to("cuda")

pred_mask, _ = sam(input_tensor, None,embed_text) 

print(pred_mask[0].shape)
np.save("pred_mask2.npy", pred_mask[0].detach().cpu().numpy())
pred_mask = np.load("pred_mask2.npy")
pred_mask = 1 / (1 + np.exp(-pred_mask))

pred_mask = np.array(pred_mask * 255., dtype=np.uint8)

cv2.imwrite("pred_mask.jpg", np.repeat(np.expand_dims(pred_mask[0], 2), 3, axis=2))

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
