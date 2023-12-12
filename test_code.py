import torch 

from models.oagan.generator import OAGAN_Generator 


gen = OAGAN_Generator(
    pretrained_encoder= "/home/tanminh/Face-Deooclusion-Pretrained/weights/r160_imintv4_statedict.pth",
    arch_encoder= "r160",
)
gen.to("cuda")
dummy_input = torch.rand(5, 3, 112, 112).to('cuda')

feat, output = gen(dummy_input)
print(output.shape)

torch.save(gen.state_dict(), "state_dict_sample.