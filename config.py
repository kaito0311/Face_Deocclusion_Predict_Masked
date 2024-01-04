class cfg:
    size_image = 112

    device = "cuda"

    noised_mask_ratio_non_occlu = 0.5
    synthetic_mask_ratio_non_occlu = 0.7
    noised_mask_ratio_occlu = 0.0
    synthetic_mask_ratio_occlu = 0.0

    # iter traning
    non_occlu_augment = 10000 
    occlu_nature = 1000

    # TM-NOTE: train of phase 1 -> only pixel loss
    GAN_LOSS_WEIGHT = 0.75
    PIXEL_LOSS_WEIGHT = 20.0
    IDENTITY_LOSS_WEIGHT = 2.5
    PERCEPTUAL_LOSS_WEIGHT = 1.0
    EDGE_LOSS_WEIGHT = 3.5
    SSIM_LOSS_WEIGHT = 2.0

    # Config
    valid_every = 1000
    print_every = 50
    batch_size = 2
    lr_gen = 1e-4
    lr_disc = 1e-4
    wd = 0.01
    START_STEP = 67000
    stage_1_iters = 0
    warmup_length = 50000  # 50k iter
    epoches = 100
    num_workers = 6
    enable_face_component_loss = True

    # TM-NOTE: Because train on phase 1 => data is only front => make model having ability gen perface face
    ROOT_DIR = "/home1/data/FFHQ/StyleGAN_data256_jpg" 
    train_data_occlu = '/home1/data/tanminh/NML-Face/list_name_file/list_name_train_masked.npy'
    train_data_non_occlu = '/home1/data/tanminh/NML-Face/list_name_file/list_name_train_no_masked.npy'
    valid_data_occlu = '/home1/data/tanminh/NML-Face/list_name_file/list_name_val_masked.npy'
    valid_data_non_occlu = '/home1/data/tanminh/NML-Face/list_name_file/list_name_val_no_masked.npy'
    training_dir = 'all_experiments/sam_training/firt_experiment'
    pretrained_g = "all_experiments/sam_training/firt_experiment/ckpt/ckpt_gen_lastest.pt"
    pretrained_d = "all_experiments/sam_training/firt_experiment/ckpt/ckpt_dis_lastest.pt"


from box import Box

config = {
    "num_devices": 1,
    "batch_size": 4,
    "num_workers": 4,
    "num_epochs": 200,
    "eval_interval": 50,
    "save_every": 250,
    "out_dir": "out/training",
    "opt": {
        "learning_rate": 8e-4,
        "weight_decay": 1e-4,
        "decay_factor": 10,
        "steps": [60000, 86666],
        "warmup_steps": 250,
    },
    "model": {
        "type": 'vit_t',
        "checkpoint": "pretrained/step-001500-f10.97-ckpt.pth",
        "freeze": {
            "image_encoder": True,
            "prompt_encoder": False,
            "mask_decoder": False,
        },
    },
    "dataset": {
        "train": {
            "root_dir": "datasets/val2017",
            "annotation_file": "datasets/annotations/instances_val2017.json"
        },
        "val": {
            "root_dir": "datasets/val2017",
            "annotation_file": "datasets/annotations/instances_val2017.json"
        }
    }
}

cfg_sam = Box(config)
