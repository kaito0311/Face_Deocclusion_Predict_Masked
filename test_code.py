import os 
import glob 
from tqdm import tqdm 
import cv2
import torch 
import numpy as np 
from PIL import Image 
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader

from config import cfg
from models.oagan.generator import OAGAN_Generator 
from dataset.dataloader import FaceRemovedMaskedDataset, FaceDataset
from face_processor_python.mfp import FaceDetector, Aligner
from models.pre_deocclusion.de_occlu_syn import FaceDeocclusionModel 
detector = FaceDetector(
    "face_processor_python/models/retinaface_mobilev3.onnx")
aligner = Aligner()

def take_mask_align(image_ori, image_size = (256, 256)):
    image_ori = cv2.imread(image_ori, cv2.IMREAD_UNCHANGED)
    image = np.zeros(shape=(512, 512, 4))
    image[128:128+256, 128: 128+256,:] = image_ori 
    image, mask = image[:, :, :3], image[:, :, 3][:, :, None]

    faceobjects = detector.DetectFace(image)

    image_align = aligner.AlignFace(image, faceobjects[0], image_size)
    mask_align = aligner.AlignFace(mask, faceobjects[0], image_size)
    mask_align = np.expand_dims(mask_align, 2)
    mask_align = np.repeat(mask_align, 3, 2)
    mask_align = np.where(mask_align > 0, 1, 0)

    return image_align, mask_align


count = 0 

dataset = "mask_face"
list_path_source = glob.glob(f"images/FaceOcc/FaceOcc/internet/{dataset}/*.png")

save_dir = "images/mask_align_retina"
print('dataset: ', dataset)
os.makedirs(save_dir, exist_ok= True)
for path in tqdm(list_path_source):
    image_align, mask_align = take_mask_align(path)
    image_base_name = str(dataset) + "_" + "image" + "_" + str(count)+ ".jpg"
    mask_base_name = str(dataset) + "_" + "mask" + "_" + str(count) + ".jpg"
    cv2.imwrite(os.path.join(save_dir, mask_base_name), mask_align)
    cv2.imwrite(os.path.join(save_dir, image_base_name), image_align)
    count += 1 

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
