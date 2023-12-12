import torch 

from models.oagan.generator import OAGAN_Generator 


gen = OAGAN_Generator(
    pretrained_encoder= "/home/tanminh/Face-Deooclusion-Pretrained/weights/r160_imintv4_statedict.pth",
    arch_encoder= "r160",
)