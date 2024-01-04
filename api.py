import torch 
import cv2 
import numpy as np 
from PIL import Image 
from torchvision import transforms as T
from models.oagan.generator import OAGAN_Generator

import gradio as gr

from face_processor_python.mfp import FaceDetector, Aligner 

id_model = -1
model = None 
def infer_face_de_occlusion(image, kind_model=1):
    global id_model, model
    kind_model = int(kind_model)
    transforms = T.Compose([
                T.Resize((112,112)),
                T.ToTensor(),
                T.Normalize(mean=[0.5], std=[0.5])
            ])
    
    image_copy = np.copy(image)

    # image = cv2.resize(image, (640, 640))
    detector = FaceDetector("face_processor_python/models/retinaface_mobilev3.onnx")
    aligner = Aligner()
    faceobjects = detector.DetectFace(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    print((int(faceobjects[0].landmark[3].x), int(faceobjects[0].landmark[3].y)))

    image = aligner.AlignFace(image, faceobjects[0])
    print(faceobjects[0].landmark[0])
    print(type(faceobjects[0].landmark[0].x))

    cv2.imwrite("input_temp.jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


    pil_image = Image.fromarray(image) 
    pil_image_original = Image.fromarray(image_copy)
    tensor = transforms(pil_image) 
    tensor_original = transforms(pil_image_original)
    tensor = torch.unsqueeze(tensor, 0)
    tensor_original = torch.unsqueeze(tensor_original, 0)
    print("tensor shape:", tensor.shape)
    
    if kind_model == 0: 
        if kind_model != id_model:
            model = OAGAN_Generator(
                pretrained_encoder="/home1/data/tanminh/NML-Face/pretrained/r160_imintv4_statedict.pth",
                pretrain_deocclu_model= "/home1/data/tanminh/Face_Deocclusion_Predict_Masked/pretrained/ckpt_gen_lastest.pt",
                freeze_deocclu_model= True
            )
            model.load_state_dict(torch.load("all_experiments/pretrained_deocclu_training/firt_experiment/ckpt/ckpt_gen_lastest.pt", map_location="cpu"))
            id_model = kind_model
    elif kind_model == 1:
        if kind_model != id_model:
            model = OAGAN_Generator(
                pretrained_encoder="/home1/data/tanminh/NML-Face/pretrained/r160_imintv4_statedict.pth",
                pretrain_deocclu_model= "/home1/data/tanminh/Face_Deocclusion_Predict_Masked/pretrained/ckpt_gen_lastest.pt",
                freeze_deocclu_model= True
            )
            model.load_state_dict(torch.load("all_experiments/pretrained_deocclu_training/second_experiment/ckpt/ckpt_gen_lastest.pt", map_location="cpu"))
            id_model = kind_model

    model.to("cpu")
    model.eval()
    
    masked, out_front, _ = model.predict(tensor.to("cpu"))
    print("masked shape: ", masked.shape)

    _, out_front_original, _ = model(tensor_original.to('cpu'))

    o_front    = np.array((out_front.detach().numpy()[0]+ 1.0)*127.5, dtype = np.uint8)
    out_front_original    = np.array((masked.detach().numpy()[0]) * 255.0, dtype = np.uint8)
    out_front_original = np.repeat(out_front_original, 3, axis=0)

    img = np.transpose(o_front, (1, 2, 0))
    img_result_original = np.transpose(out_front_original, (1,2,0))


    img = cv2.resize(img, (224,224), interpolation= cv2.INTER_CUBIC)
    img_result_original = cv2.resize(img_result_original, (224,224), interpolation= cv2.INTER_CUBIC)
    image = cv2.resize(image, (224, 224), interpolation= cv2.INTER_CUBIC)
    image_copy = cv2.resize(image_copy, (224, 224), interpolation= cv2.INTER_CUBIC)
    cv2.imwrite("temp.jpg",img)
    
    
    return np.concatenate([image, img, image_copy, img_result_original ], axis= 1)


demo = gr.Interface(
    infer_face_de_occlusion, 
    inputs=[gr.Image(), gr.Textbox(1)], 
    outputs=["image"],
    title="Face De-occlusion",
    # examples = [["data/example_api/2599067336989313.jpg", "1"], ["data/example_api/gai.jpg", 1], ["data/example_api/trecon.jpg", 1]]
    examples= []
)

demo.launch(show_tips=True, server_name='10.9.3.239', server_port=5138)

# infer_face_de_occlusion(cv2.imread("/home/data3/tanminh/NML-Face/test.jpg"))