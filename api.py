import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms as T
from models.oagan.generator import OAGAN_Generator

import gradio as gr

from face_processor_python.mfp import FaceDetector, Aligner


def infer_face_de_occlusion(image, kind_model=1):

    kind_model = int(kind_model)
    transforms = T.Compose([
        T.Resize((112, 112)),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5])
    ])
    transforms_sam = T.Compose([
        T.Resize((1024, 1024)),
        T.ToTensor(),
    ])

    image_copy = np.copy(image)

    # image = cv2.resize(image, (640, 640))
    detector = FaceDetector(
        "face_processor_python/models/retinaface_mobilev3.onnx")
    aligner = Aligner()
    faceobjects = detector.DetectFace(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    print((int(faceobjects[0].landmark[3].x),
          int(faceobjects[0].landmark[3].y)))

    image = aligner.AlignFace(image, faceobjects[0])
    print(faceobjects[0].landmark[0])
    print(type(faceobjects[0].landmark[0].x))

    cv2.imwrite("input_temp.jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    pil_image_align = Image.fromarray(image)
    pil_image_original = Image.fromarray(image_copy)
    tensor = transforms(pil_image_align)
    tensor_original = transforms(pil_image_original)
    tensor_sam = transforms_sam(pil_image_align)
    tensor = torch.unsqueeze(tensor, 0)
    tensor_original = torch.unsqueeze(tensor_original, 0)
    tensor_sam = torch.unsqueeze(tensor_sam, 0)
    print("tensor shape:", tensor.shape)
    model = OAGAN_Generator(
        pretrained_encoder=None,
        arch_encoder="r160",
        freeze_encoder=True
    )
    model.to("cuda")
    if kind_model == 0:
        model.load_state_dict(torch.load(
            "all_experiments/sam_training/firt_experiment/ckpt/ckpt_gen_backup.pt", map_location="cuda"))
    # model.predict_masked_model.to("cuda")
    model.predict_masked_model = model.predict_masked_model.to("cuda")
    model = model.to("cuda")
    model.eval()

    masked, out_front = model.predict(tensor_sam.to("cuda"), tensor.to("cuda"))
    print("masked shape: ", masked.shape)

    _, mask_predict = model(tensor_sam.to("cuda"), tensor_original.to('cuda'))

    o_front = np.array((out_front.detach().cpu().numpy()[
                       0] + 1.0)*127.5, dtype=np.uint8)
    mask_predict = np.array(
        (masked.detach().cpu().numpy()[0]) * 255.0, dtype=np.uint8)
    mask_predict = np.repeat(mask_predict, 3, axis=0)

    res_img = np.transpose(o_front, (1, 2, 0))
    mask_predict_saved = np.transpose(mask_predict, (1, 2, 0))

    res_img = cv2.resize(res_img, (224, 224), interpolation=cv2.INTER_CUBIC)
    mask_predict_saved = cv2.resize(
        mask_predict_saved, (224, 224), interpolation=cv2.INTER_CUBIC)
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_CUBIC)
    image_copy = cv2.resize(image_copy, (224, 224),
                            interpolation=cv2.INTER_CUBIC)
    cv2.imwrite("temp.jpg", res_img)

    return np.concatenate([image, res_img, image_copy, mask_predict_saved], axis=1)


demo = gr.Interface(
    infer_face_de_occlusion,
    inputs=[gr.Image(), gr.Textbox(1)],
    outputs=["image"],
    title="Face De-occlusion",
    # examples = [["data/example_api/2599067336989313.jpg", "1"], ["data/example_api/gai.jpg", 1], ["data/example_api/trecon.jpg", 1]]
    examples=[]
)

demo.launch(show_tips=True, server_name='10.9.3.239', server_port=5137)

# infer_face_de_occlusion(cv2.imread("/home/data3/tanminh/NML-Face/test.jpg"))
