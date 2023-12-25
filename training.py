import os
import time

import cv2
import torch
import mlflow
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from config import cfg
from utils.utils import cosine_lr, get_lr
from models.oagan.generator import OAGAN_Generator
from models.oagan.disciminator import Discriminator
from losses.ssim_losses import gaussian_kernel, ssim
from dataset.dataloader import FaceRemovedMaskedDataset
from losses.vgg_feature_losses import PerceptualLoss, GANLoss
from losses.losses import pixel_wise, sobel_loss, identity_loss


gan_loss_obj = GANLoss('wgan_softplus')
percep_loss_obj = PerceptualLoss(layer_weights={'conv1_2': 0.1,
                                                'conv2_2': 0.1,
                                                'conv3_4': 1,
                                                'conv4_4': 1,
                                                'conv5_4': 1,
                                                },
                                 vgg_type='vgg19',
                                 use_input_norm=True,
                                 perceptual_weight=0.1,
                                 range_norm=True).to(cfg.device)


def discriminator_loss(disc_out_rot, disc_out_front, disc_target_front):
    # Discriminator must classify correct real/fake image
    # 0 - fake, 1 - real
    B = disc_out_rot[0].size()[0]
    loss_object = torch.nn.BCEWithLogitsLoss()
    real_loss = 0
    fake_loss = 0
    for ix in range(len(disc_out_front)):
        com_rot = disc_out_rot[ix].view(B, -1)
        com_front = disc_out_front[ix].view(B, -1)
        com_target_front = disc_target_front[ix].view(B, -1)
        mul = 1.0 if ix == 0 else 0.5
        real_loss += mul * \
            loss_object(com_target_front, torch.ones_like(com_target_front))
        fake_loss += mul * (loss_object(com_rot, torch.zeros_like(com_rot)) +
                            loss_object(com_front, torch.zeros_like(com_front)))/2.0

        # real_loss += mul * gan_loss_obj(com_target_front, target_is_real = True, is_disc = True)
        # fake_loss += mul * (gan_loss_obj(com_rot, target_is_real = False, is_disc = True) + \
        #                     gan_loss_obj(com_front, target_is_real = False, is_disc = True))

    real_loss /= len(disc_out_front)
    fake_loss /= len(disc_out_front)
    return real_loss, fake_loss, real_loss + fake_loss


def generator_loss(disc_out_front, disc_out_rot, feat_occlu, out_occlu, feat_non_occlu, out_non_occlu, target_non_occlu, step=None):
    # Gan loss
    loss_object = torch.nn.BCEWithLogitsLoss()
    B = target_non_occlu.size()[0]
    gan_loss = 0
    for ix in range(len(disc_out_front)):
        com_rot = disc_out_rot[ix].view(B, -1)
        com_front = disc_out_front[ix].view(B, -1)
        mul = 1.0 if ix == 0 else 0.5
        # gan_loss += mul * GAN_LOSS_WEIGHT * \
        #         (loss_object(com_rot, torch.ones_like(com_rot)) + loss_object(com_front, torch.ones_like(com_front)))/2.0

        gan_loss += cfg.GAN_LOSS_WEIGHT * mul * \
            (gan_loss_obj(com_rot, target_is_real=True, is_disc=False)
             + gan_loss_obj(com_front, target_is_real=True, is_disc=False))

    gan_loss /= len(disc_out_front)

    # Pixel wise
    pixel_loss = cfg.PIXEL_LOSS_WEIGHT * \
        pixel_wise(out_non_occlu, target_non_occlu)

    # # Identity FIXME
    id_loss = cfg.IDENTITY_LOSS_WEIGHT * \
        identity_loss(model_feature_extraction,
                      feat_occlu, out_occlu, feat_non_occlu, out_non_occlu)
    # id_loss = 0

    # Perceptual
    perceptual_loss, _ = percep_loss_obj(out_non_occlu, target_non_occlu)
    perceptual_loss *= cfg.PERCEPTUAL_LOSS_WEIGHT

    # Edge loss
    edge_loss = cfg.EDGE_LOSS_WEIGHT * \
        sobel_loss(out_non_occlu, target_non_occlu, reduction="mean")

    # SSIM loss
    kernel = gaussian_kernel(7, sigma=1).repeat(3, 1, 1)
    kernel = kernel.to("cuda")
    ss, cs = ssim(out_non_occlu, target_non_occlu, kernel)
    ssim_loss = cfg.SSIM_LOSS_WEIGHT * (2 - torch.mean(ss) - torch.mean(cs))

    if step < cfg.stage_1_iters:
        total_loss = pixel_loss / cfg.PIXEL_LOSS_WEIGHT + 0.1*perceptual_loss / \
            cfg.PERCEPTUAL_LOSS_WEIGHT  # L1 + perceptual loss only

    else:
        total_loss = gan_loss \
            + pixel_loss \
            + id_loss \
            + perceptual_loss \
            + edge_loss \
            + ssim_loss

    return total_loss, gan_loss, pixel_loss, id_loss, perceptual_loss, edge_loss, ssim_loss


def eval(step):
    print('-'*50)
    print('Evaluating step {} ...'.format(step))
    model_generator.eval()
    counter = 0
    vis_folder = os.path.join(cfg.training_dir, "visualize", str(step))
    os.makedirs(vis_folder, exist_ok=True)

    ckpt_folder = os.path.join(cfg.training_dir, "ckpt")
    os.makedirs(ckpt_folder, exist_ok=True)
    path_backup_gen = os.path.join(ckpt_folder, "ckpt_gen_backup.pt")
    path_lastest_gen = os.path.join(ckpt_folder, "ckpt_gen_lastest.pt")

    if os.path.isfile(path_lastest_gen):
        os.system(f"mv {path_lastest_gen} {path_backup_gen}")
    torch.save(model_generator.state_dict(), path_lastest_gen)
    model_disciminator.eval()

    path_backup_disc = os.path.join(ckpt_folder, "ckpt_dis_backup.pt")
    path_lastest_disc = os.path.join(ckpt_folder, "ckpt_dis_lastest.pt")
    if os.path.isfile(path_lastest_disc):
        os.system(f"mv {path_lastest_disc} {path_backup_disc}")
    torch.save(model_disciminator.state_dict(), path_lastest_disc)

    model_disciminator.train()
    with open(os.path.join(ckpt_folder, "infor_ckpt.txt"), 'a') as file:
        file.write(str(step) + "\n")
    file.close()

    for i, batch in enumerate(val_loader):
        inputs_rot, inputs_rot_augment,  inputs_front, inputs_front_augment = batch
        inputs_rot = inputs_rot.to('cuda')
        inputs_front = inputs_front.to('cuda')
        inputs_front_augment = inputs_front_augment.to('cuda')
        inputs_rot_augment = inputs_rot_augment.to('cuda')

        with torch.no_grad():
            _, out_rot = model_generator(inputs_rot_augment)
            _, out_front = model_generator(inputs_front_augment)
            inputs_rot = inputs_rot.detach().cpu().numpy()
            inputs_front = inputs_front.detach().cpu().numpy()
            inputs_front_augment = inputs_front_augment.detach().cpu().numpy()
            inputs_rot_augment = inputs_rot_augment.detach().cpu().numpy()
            out_rot = out_rot.detach().cpu().numpy()
            out_front = out_front.detach().cpu().numpy()
        for ix in range(len(inputs_front)):
            i_front = np.array((inputs_front[ix] + 1.0)*127.5, dtype=np.uint8)
            i_front_augment = np.array(
                (inputs_front_augment[ix] + 1.0)*127.5, dtype=np.uint8)
            i_rot = np.array((inputs_rot[ix] + 1.0)*127.5, dtype=np.uint8)
            i_rot_augment = np.array(
                (inputs_rot_augment[ix] + 1.0)*127.5, dtype=np.uint8)
            o_front = np.array((out_front[ix] + 1.0)*127.5, dtype=np.uint8)
            o_rot = np.array((out_rot[ix] + 1.0)*127.5, dtype=np.uint8)

            img = np.concatenate(
                [i_front, i_front_augment, o_front, i_rot, i_rot_augment, o_rot], axis=2)
            img = np.transpose(img, (1, 2, 0))
            cv2.imwrite(os.path.join(vis_folder, "vis_{}.jpg".format(
                counter + ix)), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        counter += len(inputs_front)
    model_generator.train()


def train():
    step = cfg.START_STEP
    print("Start step: ", step)
    for epoch in range(0, cfg.epoches):

        # os.makedirs(os.path.join (training_dir, f"EPOCH{epoch}"), exist_ok = True)
        # Generator loss
        avg_gen_loss = 0
        avg_gan_loss = 0
        avg_pixel_loss = 0
        avg_identity_loss = 0
        avg_perceptual_loss = 0
        avg_edge_loss = 0
        avg_ssim_loss = 0
        # Discriminator loss
        avg_real_loss = 0
        avg_fake_loss = 0
        avg_disc_loss = 0
        print("EPOCH : " + str(epoch))

        for i, batch in enumerate(train_loader):
            if len(batch[0]) != cfg.batch_size:
                print('[WARNING] Skipped batch {} due to invalid number/batch_size:'.format(
                    i), len(batch[0]), cfg.batch_size)
                continue

            step += 1
            lr = scheduler_gen(step)
            _ = scheduler_disc(step)

            # torch.autograd.set_detect_anomaly(True)
            i0 = time.time()
            for p in model_disciminator.parameters():
                p.requires_grad = False
            # Optimize generator
            optimizer_gen.zero_grad()
            inputs_occlu, input_occlu_augment, inputs_non_occlu, inputs_non_occlu_augment = batch
            inputs_occlu = inputs_occlu.to('cuda')
            inputs_non_occlu = inputs_non_occlu.to('cuda')
            inputs_non_occlu_augment = inputs_non_occlu_augment.to('cuda')
            input_occlu_augment = input_occlu_augment.to("cuda")

            # Get generator output
            _, out_occlu = model_generator(input_occlu_augment)
            feat_occlu = model_feature_extraction(inputs_occlu)[0]
            _, out_non_occlu = model_generator(inputs_non_occlu_augment)
            feat_non_occlu = model_feature_extraction(inputs_non_occlu)[0]
            # Get disciminator output
            disc_occlu = model_disciminator(out_occlu)
            disc_non_occlu = model_disciminator(out_non_occlu)
            # Get generator loss
            total_loss, gan_loss, pixel_loss, identity_loss, perceptual_loss, edge_loss, ssim_loss = generator_loss(disc_non_occlu,
                                                                                                                    disc_occlu,
                                                                                                                    feat_occlu,
                                                                                                                    out_occlu,
                                                                                                                    feat_non_occlu,
                                                                                                                    out_non_occlu,
                                                                                                                    inputs_non_occlu,
                                                                                                                    step=step)
            total_loss.backward()
            optimizer_gen.step()
            i1 = time.time()

            # Optimize discriminator
            for p in model_disciminator.parameters():
                p.requires_grad = True
            optimizer_disc.zero_grad()
            # .detach() here mean disable gradient from generator
            fake_out_rot = model_disciminator(out_occlu.detach())
            # .detach() here mean disable gradient from generator
            fake_out_front = model_disciminator(out_non_occlu.detach())
            real_d_pred = model_disciminator(inputs_non_occlu)
            real_loss, fake_loss, disc_loss = discriminator_loss(
                fake_out_rot, fake_out_front, real_d_pred)
            disc_loss.backward()
            # torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer_disc.step()
            i2 = time.time()
            avg_gen_loss += total_loss.detach().cpu().numpy()
            avg_gan_loss += gan_loss.detach().cpu().numpy()
            avg_pixel_loss += pixel_loss.detach().cpu().numpy()
            avg_identity_loss += identity_loss.detach().cpu().numpy()
            avg_perceptual_loss += perceptual_loss.detach().cpu().numpy()
            avg_ssim_loss += ssim_loss.detach().cpu().numpy()
            avg_edge_loss += edge_loss.detach().cpu().numpy()
            avg_real_loss += real_loss.detach().cpu().numpy()
            avg_fake_loss += fake_loss.detach().cpu().numpy()
            avg_disc_loss += disc_loss.detach().cpu().numpy()

            i3 = time.time()
            if step % cfg.print_every == 0 and step > 1:
                print('-'*50)
                tavg_gen_loss = avg_gen_loss/(i + 1)
                tavg_gan_loss = avg_gan_loss/(i + 1)
                tavg_pixel_loss = avg_pixel_loss/(i + 1)
                tavg_identity_loss = avg_identity_loss/(i + 1)
                tavg_edge_loss = avg_edge_loss / (i + 1)
                tavg_perceptual_loss = avg_perceptual_loss/(i + 1)
                tavg_real_loss = avg_real_loss/(i + 1)
                tavg_fake_loss = avg_fake_loss/(i + 1)
                tavg_disc_loss = avg_disc_loss/(i + 1)
                tavg_ssim_loss = avg_ssim_loss/(i + 1)

                print('step', step)

                print('avg_gan_loss = ', tavg_gan_loss)
                print('avg_pixel_loss = ', tavg_pixel_loss)
                print('avg_identity_loss = ', tavg_identity_loss)
                print('avg_perceptual_loss = ', tavg_perceptual_loss)
                print("avg_edge_loss = ", tavg_edge_loss)
                print("avg_ssim_loss = ", tavg_ssim_loss)
                print('avg_real_loss = ', tavg_real_loss)
                print('avg_fake_loss = ', tavg_fake_loss)
                print('*********************')
                print('avg_gen_loss = ', tavg_gen_loss)
                print('avg_disc_loss = ', tavg_disc_loss)

                print('lr_gen = ', get_lr(optimizer_gen))
                print('lr_disc = ', get_lr(optimizer_disc))
                print('time data / time gen / time disc / time all = {} / {} / {} / {}'.format(
                    i0 - i4, i1-i0, i2-i1, i3-i0))

                metrics = dict(
                    tavg_gan_loss=tavg_gan_loss,
                    tavg_pixel_loss=tavg_pixel_loss,
                    tavg_identity_loss=tavg_identity_loss,
                    tavg_perceptual_loss=tavg_perceptual_loss,
                    tavg_edge_loss=tavg_edge_loss,
                    tavg_ssim_loss=tavg_ssim_loss,
                    tavg_real_loss=tavg_real_loss,
                    tavg_fake_loss=tavg_fake_loss,
                    tavg_gen_loss=tavg_gen_loss,
                    tavg_disc_loss=tavg_disc_loss,
                    optimizer_gen=get_lr(optimizer_gen),
                    optimizer_disc=get_lr(optimizer_disc)
                )

                mlflow.log_metrics(metrics=metrics, step=step)

            if step % cfg.valid_every == 0 and step > 0:
                print('VALIDATE')
                eval(step)

                # os.makedirs(os.path.join(training_dir, 'weights'), exist_ok = True)
                # torch.save(model_gen.state_dict(), os.path.join(training_dir, 'weights', 'checkpoint_g_{}.pth'.format(step)))
                # torch.save(model_disc.state_dict(), os.path.join(training_dir, 'weights', 'checkpoint_d_{}.pth'.format(step)))
            # if step % 1000 == 0:
            #     time.sleep(60)
            i4 = time.time()

        avg_gen_loss /= (i + 1)
        avg_gan_loss /= (i + 1)
        avg_pixel_loss /= (i + 1)
        avg_edge_loss /= (i + 1)
        avg_identity_loss /= (i + 1)
        avg_perceptual_loss /= (i + 1)
        avg_real_loss /= (i + 1)
        avg_fake_loss /= (i + 1)
        avg_disc_loss /= (i + 1)

        print('avg_gan_loss = ', avg_gan_loss)
        print('avg_pixel_loss = ', avg_pixel_loss)
        print('avg_identity_loss = ', avg_identity_loss)
        print('avg_perceptual_loss = ', avg_perceptual_loss)
        print('avg_edge_loss = ', avg_edge_loss)
        print('avg_real_loss = ', avg_real_loss)
        print('avg_fake_loss = ', avg_fake_loss)
        print('*********************')
        print('avg_gen_loss = ', avg_gen_loss)
        print('avg_disc_loss = ', avg_disc_loss)


if __name__ == "__main__":
    # Init mlflow
    experiment_name = 'Face-Deoclusion-Predict-Masked'
    experiment = mlflow.set_experiment(experiment_name=experiment_name)
    run = mlflow.start_run(run_name="Phase 1",
                           run_id=None,
                           experiment_id=experiment.experiment_id,
                           description="Init")

    ''' Define the model '''
    from models.backbones.imintv5 import iresnet160
    model_feature_extraction = iresnet160(pretrained=False)
    model_feature_extraction.load_state_dict(
        torch.load(
            "/home1/data/tanminh/NML-Face/pretrained/r160_imintv4_statedict.pth")
    )

    model_generator = OAGAN_Generator(
        pretrained_encoder="/home1/data/tanminh/NML-Face/pretrained/r160_imintv4_statedict.pth",
        arch_encoder="r160",
        freeze_encoder= True
    )
    model_disciminator = Discriminator(
        input_size=cfg.size_image, enable_face_component_loss=cfg.enable_face_component_loss)

    if cfg.pretrained_g is not None: 
        print("[INFO]: Loadding pretrained ", cfg.pretrained_g)
        print("[INFO]: Loadding pretrained ", cfg.pretrained_d)
        model_generator.load_state_dict(torch.load(cfg.pretrained_g))
        model_disciminator.load_state_dict(torch.load(cfg.pretrained_d))


    model_generator.to(cfg.device)
    model_disciminator.to(cfg.device)
    model_feature_extraction.to(cfg.device)
    model_generator.train()
    model_disciminator.train()
    model_feature_extraction.eval()

    ''' Define dataloader '''
    trainset = FaceRemovedMaskedDataset(
        list_name_data_occlusion=cfg.train_data_non_occlu,  # NOTE
        list_name_data_non_occlusion=cfg.train_data_non_occlu,
        root_dir=cfg.ROOT_DIR,
        is_train=True,
        path_occlusion_object="images/occlusion_object/clean_segment",
    )
    # trainset = FaceRemovedMaskedDataset(
    #     list_name_data_occlusion=cfg.train_data_occlu,
    #     list_name_data_non_occlusion=cfg.train_data_non_occlu,
    #     root_dir=cfg.ROOT_DIR,
    #     is_train=True,
    #     path_occlusion_object="/home1/data/tanminh/NML-Face/khautrang",
    # )
    valset = FaceRemovedMaskedDataset(
        list_name_data_occlusion=cfg.valid_data_occlu,
        list_name_data_non_occlusion=cfg.valid_data_non_occlu,
        root_dir=cfg.ROOT_DIR,
        is_train=False,
        path_occlusion_object="images/occlusion_object/clean_segment",
    )
    train_loader = DataLoader(
        trainset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, drop_last=True)

    val_loader = DataLoader(
        valset, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, drop_last=True
    )
    print('  - Total train:', len(trainset))
    print('  - Total val:', len(valset))
    num_batches = len(train_loader)

    ''' Optimizer & Scheduler '''
    params_gen = []
    for name, p in model_generator.named_parameters():
        params_gen.append(p)

    print("[INFO] number trained params gen: ", len(params_gen))
    params_disc = [p for name, p in model_generator.named_parameters()]
    optimizer_gen = torch.optim.AdamW(
        params_gen, lr=cfg.lr_gen, weight_decay=cfg.wd)
    optimizer_disc = torch.optim.AdamW(
        params_disc, lr=cfg.lr_disc, weight_decay=cfg.wd)

    scheduler_gen = cosine_lr(optimizer_gen, cfg.lr_gen,
                              cfg.warmup_length, cfg.epoches * num_batches)
    scheduler_disc = cosine_lr(
        optimizer_disc, cfg.lr_disc, cfg.warmup_length, cfg.epoches * num_batches)
    ''' Training'''

    train()
