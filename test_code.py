import os 

import torch 
import numpy as np 

from models.oagan.generator import OAGAN_Generator 

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
