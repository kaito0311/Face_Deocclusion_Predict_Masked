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





kind_model = 40000
model = OAGAN_Generator(
    pretrained_encoder="/home1/data/tanminh/NML-Face/pretrained/r160_imintv4_statedict.pth",
    pretrain_deocclu_model= "/home1/data/tanminh/Face_Deocclusion_Predict_Masked/pretrained/ckpt_gen_lastest.pt",
    freeze_deocclu_model= True
)
model.load_state_dict(torch.load(f"all_experiments/pretrained_deocclu_training/second_experiment/ckpt/ckpt_gen_{kind_model}.pt", map_location="cpu"))

model.eval()
model.cpu() 

batch_size=1
x = torch.randn(batch_size, 3, 112, 112, requires_grad=True)
torch_out = model.predict_masked_model(x)

# Export the model
torch.onnx.export(model.predict_masked_model,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  "only_predict_mask_40k.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=15,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}})



exit()

detector = FaceDetector(
    "face_processor_python/models/retinaface_mobilev3.onnx")
aligner = Aligner()



source_dir = "images/mask_align_retina"

list_path = glob.glob(source_dir + "/*.jpg")

save_dir_mask = "/home1/data/tanminh/MaskTheFace/data/mask"
save_dir_image = "/home1/data/tanminh/MaskTheFace/data/image"

os.makedirs(save_dir_mask, exist_ok= True)
os.makedirs(save_dir_image, exist_ok= True)


# image = "images/mask_align_retina/CelebAHQ_image_4681.jpg"
# mask = "images/mask_align_retina/CelebAHQ_mask_17416.jpg"
# mask = cv2.imread(mask)
# mask = mask * 255.0
# mask = np.clip(mask, 0, 255)
# mask = np.array(mask, np.uint8)
# cv2.imwrite("mask.jpg", mask)

# # exit()
# count = 0 

# for path in tqdm(list_path):
#     is_image = os.path.basename(path).split("_")[-2] == "image" 
    
#     if is_image: 
#         image = cv2.imread(path) 

#         path_mask = os.path.basename(path).split("_")
#         path_mask[-2] = "mask" 
#         path_mask = "_".join(path_mask)
#         path_mask = os.path.join(source_dir, path_mask)

#         if not os.path.isfile(path_mask):
#             print("[ERROR] not found ", path_mask)
#             continue

#         mask = cv2.imread(path_mask)

#         mask = mask * 255.0 
#         mask = np.clip(mask, 0, 255)
#         mask = np.array(mask, np.uint8)

        
#         cv2.imwrite(os.path.join(save_dir_image, str(count) + ".jpg"), image)
#         cv2.imwrite(os.path.join(save_dir_mask, str(count) + ".jpg"), mask)

#         count += 1
        
    

    
#     # os.system(cmd) 






# exit()
def take_mask_align_from_png(image_ori, image_size = (256, 256)):
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

def take_mask_align(image, mask, image_size = (256, 256)):
    image = cv2.imread(image) 
    mask = cv2.imread(mask) 

    faceobjects = detector.DetectFace(image)
    
    if len(faceobjects) == 0:
        return [], [] 

    list_face, list_mask = [], [] 

    for faceobj in faceobjects:
        image_align = aligner.AlignFace(image, faceobj, image_size)
        mask_align = aligner.AlignFace(mask, faceobj, image_size)
        list_face.append(image_align.copy())
        list_mask.append(mask_align.copy())
    
    return list_face, list_mask
    



count = 0 

root_dir = "images/segment_entire/"

save_image = "images/segment_entire/align_image/"
save_mask = "images/segment_entire/align_mask"
os.makedirs(save_image, exist_ok= True) 
os.makedirs(save_mask, exist_ok= True) 


for path in tqdm(os.listdir(os.path.join(root_dir, "image"))):
    list_image, list_mask = take_mask_align(os.path.join(root_dir, "image", path), os.path.join(root_dir, "mask", path))

    idx = 0 
    for image_align, mask_align in zip(list_image, list_mask):
        base_name = str(idx) + "_" + path 
        cv2.imwrite(os.path.join(save_image, base_name), image_align)
        cv2.imwrite(os.path.join(save_mask, base_name), mask_align)
        idx+=1 


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
