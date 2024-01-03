import os 

import cv2
import torch 
import numpy as np 
from torch.utils.data import Dataset, DataLoader

from config import cfg, cfg_sam
from models.oagan.generator import OAGAN_Generator 
from dataset.dataloader import FaceRemovedMaskedDataset, FaceDataset
from face_processor_python.mfp import FaceDetector, Aligner

from model_sam import Model 



import sys
sys.path.append("..")
from mobile_sam import sam_model_registry, SamPredictor
import matplotlib.pyplot as plt 




def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    


sam_checkpoint = "pretrained/mobile_sam.pt"
model_type = "vit_t"

device = "cuda" if torch.cuda.is_available() else "cpu"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
sam.eval()

predictor = SamPredictor(sam)

image = cv2.imread('images/test/images (3).jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# detect and align face 
detector = FaceDetector(
    "face_processor_python/models/retinaface_mobilev3.onnx")
aligner = Aligner()
faceobjects = detector.DetectFace(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
print((int(faceobjects[0].landmark[3].x),
        int(faceobjects[0].landmark[3].y)))

image = aligner.AlignFace(image, faceobjects[0])
image = cv2.resize(image, (256, 256))


predictor.set_image(image)

# input_box = np.array([30, 78, 230, 158])
# input_box = np.array([[30, 0, 230, 128]])

input_boxes = torch.tensor([[0, 128, 256, 256], 
                            [30, 0, 230, 128], 
                            [30, 78, 230, 158]])
input_boxes = input_boxes
transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])
print(input_boxes.shape)
# masks, _, _ = predictor.predict(
#     point_coords=None,
#     point_labels=None,
#     box=input_boxes,
#     multimask_output=False,
# )
masks, scores, _ = predictor.predict_torch(
    point_coords=None,
    point_labels=None,
    boxes=transformed_boxes,
    multimask_output=False,
)

print(scores)
plt.figure(figsize=(10, 10))
plt.imshow(image)
for mask in masks:
    show_mask(mask[0], plt.gca())
for box in input_boxes:
    show_box(box, plt.gca())
plt.axis('off')
plt.savefig("sample.jpg")


exit()
sam = Model(cfg_sam)
sam.setup() 
sam.to("cuda")
sam.eval() 

embed_text = torch.from_numpy(np.load("pretrained/feature_text.npy"))
embed_text = embed_text.to("cuda")


image = cv2.imread("images/val/masked/100000166523059_face_3.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

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
