import os 
import glob 
import cv2
import torch 
import numpy as np 
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm 
from config import cfg, cfg_sam
from models.oagan.generator import OAGAN_Generator 
from face_processor_python.mfp import FaceDetector, Aligner
from dataset.dataloader import FaceRemovedMaskedDataset, FaceDataset

from model_sam import Model 

detector = FaceDetector(
    "face_processor_python/models/retinaface_mobilev3.onnx")
aligner = Aligner()

image = cv2.imread("images/FaceOcc/FaceOcc/internet/glasses/0.png")
cv2.imwrite("test_face.jpg", image)


# face = cv2.imread("images/val/masked/100000158402433_face_0.jpg")
# occlu = cv2.imread("images/mask_align_retina/ffhq_image_17614.jpg")
# mask = cv2.imread("images/mask_align_retina/ffhq_mask_17614.jpg")

# face = (1-mask) * face + mask * occlu

# face = cv2.blur(face,(3,3))
# # kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
# # face = cv2.filter2D(face, -1, kernel)
# face = cv2.resize(face, (112,112))
# cv2.imwrite("test_face.jpg", face)



exit()

# spectu = "images/FaceOcc/FaceOcc/CelebAHQ/52.png"
# face = "images/val/masked/100000139670028_face_2.jpg"
# spectu = cv2.imread(spectu, cv2.IMREAD_UNCHANGED)
# face = cv2.imread(face) 
# mask, spectu = spectu[:, :, 3], spectu[:, :, :3]
# face = face * (1 - mask)[:, :, None] + mask[:, :, None] * spectu 
# cv2.imwrite("test_face.jpg", face)

# face = cv2.imread("images/FaceOcc/FaceOcc/CelebAHQ/52.png")
# mask = cv2.imread("images/FaceOcc/FaceOcc/internet/glasses/44.png")

# face = np.where(mask != 0, mask, face)

# cv2.imwrite("test_face.jpg", face)

# exit()


# image = cv2.imread("images/FaceOcc/FaceOcc/ffhq/68597.png", cv2.IMREAD_UNCHANGED)
# print(image.shape)



# exit()

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
    image_base_name = str(dataset) + "_" + "image" + "_" + os.path.basename(path).split(".")[0] + ".jpg"
    mask_base_name = str(dataset) + "_" + "mask" + "_" + os.path.basename(path).split(".")[0] + ".jpg"
    cv2.imwrite(os.path.join(save_dir, mask_base_name), mask_align)
    cv2.imwrite(os.path.join(save_dir, image_base_name), image_align)
    count += 1 






# def convert(image_ori, image_face, image_size = (256, 256)):
#     global detector, aligner 

#     image = np.zeros(shape=(512, 512, 4))
#     image_ori = cv2.imread(image_ori, cv2.IMREAD_UNCHANGED)
#     image_face= cv2.imread(image_face) 

#     image[128:128+256, 128: 128+256,:] = image_ori 
#     image, mask = image[:, :, :3], image[:, :, 3][:, :, None]

#     faceobjects = detector.DetectFace(image)

#     image_align = aligner.AlignFace(image, faceobjects[0], image_size)
#     mask_align = aligner.AlignFace(mask, faceobjects[0], image_size)
#     mask_align = np.expand_dims(mask_align, 2)
#     mask_align = np.repeat(mask_align, 3, 2)
#     print(mask_align.shape)

#     image_face = cv2.resize(image_face, image_size)
#     mask_align = np.where(mask_align > 0, 1, 0)
#     image_face = image_face * (1 - mask_align) + image_align * (mask_align)
    
#     image_align = image_align * mask_align


exit() 

convert(
    image_face="images/val/masked/100000139670028_face_2.jpg",
    image_ori = "images/FaceOcc/FaceOcc/CelebAHQ/52.png", 
)



exit() 


# exit()




spectu = "images/FaceOcc/FaceOcc/CelebAHQ/52.png"
face = "images/val/masked/100000139670028_face_2.jpg"

image = np.zeros(shape=(512, 512, 3))

spectu = cv2.imread(spectu)
image[128:128+256, 128: 128+256,:] = spectu 
spectu = image
print(spectu.shape)

faceobjects = detector.DetectFace(spectu)
print((int(faceobjects[0].landmark[3].x),
        int(faceobjects[0].landmark[3].y)))
spectu = cv2.circle(spectu, (int(faceobjects[0].landmark[0].x),int(faceobjects[0].landmark[0].y)), 2, (255, 0, 0), -1)
spectu = cv2.circle(spectu, (int(faceobjects[0].landmark[1].x),int(faceobjects[0].landmark[1].y)), 2, (255, 0, 0), -1)
spectu = cv2.circle(spectu, (int(faceobjects[0].landmark[2].x),int(faceobjects[0].landmark[2].y)), 2, (255, 0, 0), -1)
spectu = cv2.circle(spectu, (int(faceobjects[0].landmark[3].x),int(faceobjects[0].landmark[3].y)), 2, (255, 0, 0), -1)
spectu = cv2.circle(spectu, (int(faceobjects[0].landmark[4].x),int(faceobjects[0].landmark[4].y)), 2, (255, 0, 0), -1)
# face = cv2.imread(face)
spectu = aligner.AlignFace(spectu, faceobjects[0])
print(spectu.shape)

# face = np.where(spectu != 0, spectu, face)
# print(spectu)

cv2.imwrite("test_face.jpg", spectu) 
cv2.imwrite("test_face.jpg", spectu) 




exit() 

count = 0
for i in os.listdir("images/occlusion_object/clean_segment"):
    if "hat" in i:
        count += 1
print(count)


exit()
weight = torch.load("all_experiments/sam_training/firt_experiment/ckpt/ckpt_gen_lastest.pt")
print(weight.keys())


exit()
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
