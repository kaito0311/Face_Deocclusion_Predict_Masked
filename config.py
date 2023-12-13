class cfg:
    size_image = 112

    device = "cuda"

    noised_mask_ratio_non_occlu = 0.7
    noised_mask_ratio_occlu = 0.0
    synthetic_mask_ratio_occlu = 0.0
    synthetic_mask_ratio_non_occlu = 0.0

    # TM-NOTE: train of phase 1 -> only pixel loss
    GAN_LOSS_WEIGHT = 0.75
    PIXEL_LOSS_WEIGHT = 20.0
    IDENTITY_LOSS_WEIGHT = 2.5
    PERCEPTUAL_LOSS_WEIGHT = 1.0
    EDGE_LOSS_WEIGHT = 3.5
    SSIM_LOSS_WEIGHT = 2.0

    # Config
    valid_every = 100
    print_every = 50
    batch_size = 4
    lr_gen = 1e-4
    lr_disc = 1e-4
    wd = 0.01
    START_STEP = 0
    stage_1_iters = 500000
    warmup_length = 50000  # 50k iter
    epoches = 100000
    num_workers = 6
    enable_face_component_loss = True
    SKIP_ATTENTION = False
    FREEZE_DECODE = True

    # TM-NOTE: Because train on phase 1 => data is only front => make model having ability gen perface face
    ROOT_DIR = "/home1/data/FFHQ/StyleGAN_data256_jpg" 
    train_data_occlu = '/home1/data/tanminh/NML-Face/list_name_file/list_name_train_no_masked.npy'
    train_data_non_occlu = '/home1/data/tanminh/NML-Face/list_name_file/list_name_train_no_masked.npy'
    valid_data_occlu = '/home1/data/tanminh/NML-Face/list_name_file/list_name_val_masked.npy'
    valid_data_non_occlu = '/home1/data/tanminh/NML-Face/list_name_file/list_name_val_no_masked.npy'
    training_dir = 'experiment_224'
    pretrained_g = "experiment_224/ckpt/ckpt_gen_lastest.pt"
    pretrained_d = "experiment_224/ckpt/ckpt_dis_lastest.pt"
