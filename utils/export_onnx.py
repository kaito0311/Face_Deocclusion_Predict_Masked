import torch 

from models.oagan.generator import OAGAN_Generator




kind_model = 40000
model = OAGAN_Generator(
    pretrained_encoder=None,
    pretrain_deocclu_model= None,
    freeze_deocclu_model= True
)
model.load_state_dict(torch.load(f"all_experiments/pretrained_deocclu_training/second_experiment/ckpt/ckpt_gen_{kind_model}.pt", map_location="cpu"))

model.eval()
model.cpu() 

batch_size=1
x = torch.randn(batch_size, 3, 112, 112, requires_grad=True)
torch_out = model(x)

# Export the model
torch.onnx.export(model,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  "predict_mask_40k_2.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=15,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}})
