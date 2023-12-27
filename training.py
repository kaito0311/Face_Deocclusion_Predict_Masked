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
from dataset.dataloader import FaceDataset
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


def discriminator_loss(disc_out_res, disc_real_image):
    # Discriminator must classify correct real/fake image
    # 0 - fake, 1 - real
    B = disc_out_res[0].size()[0]
    loss_object = torch.nn.BCEWithLogitsLoss()
    real_loss = 0
    fake_loss = 0
    for ix in range(len(disc_out_res)):
        com_restore = disc_out_res[ix].view(B, -1)
        com_target_real = disc_real_image[ix].view(B, -1)
        mul = 1.0 if ix == 0 else 0.5
        # real_loss += mul * \
        #     loss_object(com_target_front, torch.ones_like(com_target_front))
        # fake_loss += mul * (loss_object(com_restore, torch.zeros_like(com_restore)))

        real_loss += mul * \
            gan_loss_obj(com_target_real, target_is_real=True, is_disc=True)
        fake_loss += mul * \
            (gan_loss_obj(com_restore, target_is_real=False, is_disc=True))

    real_loss /= len(disc_out_res)
    fake_loss /= len(disc_out_res)
    return real_loss, fake_loss, real_loss + fake_loss


def generator_loss(disc_restore, restore_image, ori_image, mask=None, is_batch_occlu=False, step=None):
    # Gan loss
    loss_object = torch.nn.BCEWithLogitsLoss()
    B = ori_image.size()[0]
    gan_loss = 0
    for ix in range(len(disc_restore)):
        com_restore = disc_restore[ix].view(B, -1)
        mul = 1.0 if ix == 0 else 0.5
        # gan_loss += mul * GAN_LOSS_WEIGHT * \
        #         (loss_object(com_rot, torch.ones_like(com_rot)) + loss_object(com_front, torch.ones_like(com_front)))/2.0

        gan_loss += cfg.GAN_LOSS_WEIGHT * mul * \
            (gan_loss_obj(com_restore, target_is_real=True, is_disc=False))

    gan_loss /= len(disc_restore)

    # Pixel wise
    pixel_loss = cfg.PIXEL_LOSS_WEIGHT * \
        pixel_wise(restore_image, ori_image)

    # # Identity FIXME
    id_loss = cfg.IDENTITY_LOSS_WEIGHT * \
        identity_loss(model_feature_extraction,
                      restore_image=restore_image, ori_image=ori_image)
    # id_loss = 0

    # Perceptual
    perceptual_loss, _ = percep_loss_obj(restore_image, ori_image)
    perceptual_loss *= cfg.PERCEPTUAL_LOSS_WEIGHT

    # Edge loss
    edge_loss = cfg.EDGE_LOSS_WEIGHT * \
        sobel_loss(restore_image, ori_image, reduction="mean")

    # SSIM loss
    kernel = gaussian_kernel(7, sigma=1).repeat(3, 1, 1)
    kernel = kernel.to("cuda")
    ss, cs = ssim(restore_image, ori_image, kernel)
    ssim_loss = cfg.SSIM_LOSS_WEIGHT * (2 - torch.mean(ss) - torch.mean(cs))

    if is_batch_occlu:
        total_loss = pixel_loss / cfg.PIXEL_LOSS_WEIGHT + 0.1*perceptual_loss / \
            cfg.PERCEPTUAL_LOSS_WEIGHT + gan_loss  # L1 + perceptual loss only

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


    for i, batch in enumerate(non_occlu_val_loader):
        mask, augment_image, ori_image = batch
        mask = mask.to(cfg.device) 
        augment_image = augment_image.to(cfg.device) 
        ori_image = ori_image.to(cfg.device)


        with torch.no_grad():
            mask_predict, out_rot = model_generator.predict(augment_image)
            mask = mask.detach().cpu().numpy() 
            mask_predict = mask_predict.detach().cpu().numpy() 
            mask_predict = np.repeat(mask_predict, 3, axis=1)
            augment_image = augment_image.detach().cpu().numpy() 
            ori_image = ori_image.detach().cpu().numpy()
            restore_image = out_rot.detach().cpu().numpy()  

        for idx in range(len(augment_image)):
            save_augment = np.array(127.5 * (augment_image[idx] + 1.0), dtype= np.uint8)
            save_res = np.array(127.5 * (restore_image[idx] + 1.0), dtype= np.uint8)
            save_ori = np.array(127.5 * (ori_image[idx] + 1.0), dtype= np.uint8)
            save_mask= np.array(255. * mask[idx], dtype= np.uint8)
            save_mask_predict = np.array(255. * mask_predict[idx], dtype= np.uint8)
            img = np.concatenate(
                [save_ori, save_augment, save_res, save_mask, save_mask_predict], axis=2)
            img = np.transpose(img, (1, 2, 0))
            cv2.imwrite(os.path.join(vis_folder, "vis_{}.jpg".format(
                counter + idx)), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        counter += len(ori_image)
        
    for i, batch in enumerate(occlu_val_loader):
        mask, augment_image, ori_image = batch
        mask = mask.to(cfg.device) 
        augment_image = augment_image.to(cfg.device) 
        ori_image = ori_image.to(cfg.device)


        with torch.no_grad():
            mask_predict, out_rot = model_generator.predict(augment_image)
            mask = mask.detach().cpu().numpy() 
            mask_predict = mask_predict.detach().cpu().numpy() 
            mask_predict = np.repeat(mask_predict, 3, axis=1)
            augment_image = augment_image.detach().cpu().numpy() 
            ori_image = ori_image.detach().cpu().numpy()
            restore_image = out_rot.detach().cpu().numpy()  

        for idx in range(len(augment_image)):
            save_augment = np.array(127.5 * (augment_image[idx] + 1.0), dtype= np.uint8)
            save_res = np.array(127.5 * (restore_image[idx] + 1.0), dtype= np.uint8)
            save_ori = np.array(127.5 * (ori_image[idx] + 1.0), dtype= np.uint8)
            save_mask= np.array(255. * mask[idx], dtype= np.uint8)
            save_mask_predict = np.array(255. * mask_predict[idx], dtype= np.uint8)
            img = np.concatenate(
                [save_ori, save_augment, save_res, save_mask, save_mask_predict], axis=2)
            img = np.transpose(img, (1, 2, 0))
            cv2.imwrite(os.path.join(vis_folder, "vis_{}.jpg".format(
                counter + idx)), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        counter += len(ori_image)

    model_generator.train()


def train():
    step = cfg.START_STEP
    is_batch_occlu = False
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

        i = 0
        non_occlu_train_iter = iter(non_occlu_train_loader)
        occlu_train_iter = iter(occlu_train_loader)

        while True:
            # Define batch
            try:
                i += 1 
                if is_batch_occlu:
                    batch = next(occlu_train_iter)
                else:
                    batch = next(non_occlu_train_iter)
            except StopIteration:
                break

            if len(batch[0]) != cfg.batch_size:
                print('[WARNING] Skipped batch {} due to invalid number/batch_size:'.format(
                    i), len(batch[0]), cfg.batch_size)
                continue

            step += 1

            if is_batch_occlu: 
                if cfg.occlu_nature > 0 and step % cfg.occlu_nature == 0: 
                    is_batch_occlu = not is_batch_occlu 
            elif not is_batch_occlu: 
                if cfg.non_occlu_augment > 0 and step % cfg.non_occlu_augment == 0: 
                    is_batch_occlu = not is_batch_occlu

            lr = scheduler_gen(step)
            _ = scheduler_disc(step)

            # torch.autograd.set_detect_anomaly(True)
            i0 = time.time()
            for p in model_disciminator.parameters():
                p.requires_grad = False

            # Optimize generator
            optimizer_gen.zero_grad()

            mask, augment_image, ori_image = batch
            mask = mask.to(cfg.device)
            augment_image = augment_image.to(cfg.device)
            ori_image = ori_image.to(cfg.device)

            # Get generator output
            _, out_restore = model_generator(augment_image)
            # Get disciminator output
            disc_restore = model_disciminator(out_restore)
            # Get generator loss
            total_loss, gan_loss, pixel_loss, identity_loss, perceptual_loss, edge_loss, ssim_loss = generator_loss(disc_restore=disc_restore,
                                                                                                                    restore_image=out_restore,
                                                                                                                    ori_image=ori_image,
                                                                                                                    mask=None,
                                                                                                                    is_batch_occlu=is_batch_occlu,
                                                                                                                    step=step)
            total_loss.backward()
            optimizer_gen.step()
            i1 = time.time()

            # Optimize discriminator
            for p in model_disciminator.parameters():
                p.requires_grad = True
            optimizer_disc.zero_grad()
            # .detach() here mean disable gradient from generator
            fake_out_res = model_disciminator(out_restore.detach())
            # .detach() here mean disable gradient from generator
            real_d_pred = model_disciminator(ori_image)
            real_loss, fake_loss, disc_loss = discriminator_loss(
                fake_out_res, real_d_pred)
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
    experiment_name = 'Face-Deoclusion-Predict-Masked-Ver2'
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
        freeze_encoder=True
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
    non_occlu_trainset = FaceDataset(
        path_list_name_data=cfg.train_data_non_occlu,
        root_dir=cfg.ROOT_DIR,
        ratio_occlu=cfg.synthetic_mask_ratio_non_occlu,
        is_train=True,
        path_occlusion_object="images/occlusion_object/clean_segment"
    )

    occlu_trainset = FaceDataset(
        path_list_name_data=cfg.train_data_occlu,
        root_dir=cfg.ROOT_DIR,
        ratio_occlu=cfg.synthetic_mask_ratio_occlu,
        is_train=True,
        path_occlusion_object="images/occlusion_object/clean_segment"
    )

    non_occlu_valset = FaceDataset(
        path_list_name_data=cfg.valid_data_non_occlu,
        root_dir=cfg.ROOT_DIR,
        ratio_occlu=cfg.synthetic_mask_ratio_non_occlu,
        is_train=False,
        path_occlusion_object="images/occlusion_object/clean_segment"
    )

    occlu_valset = FaceDataset(
        path_list_name_data=cfg.valid_data_occlu,
        root_dir=cfg.ROOT_DIR,
        ratio_occlu=cfg.synthetic_mask_ratio_occlu,
        is_train=False,
        path_occlusion_object="images/occlusion_object/clean_segment"
    )

    non_occlu_train_loader = DataLoader(
        non_occlu_trainset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, drop_last=True)
    occlu_train_loader = DataLoader(
        occlu_trainset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, drop_last=True)
    non_occlu_val_loader = DataLoader(
        non_occlu_valset, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, drop_last=True)
    occlu_val_loader = DataLoader(
        occlu_valset, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, drop_last=True)

    print('  - Total train non-occlu:', len(non_occlu_train_loader))
    print('  - Total train occlusion:', len(occlu_train_loader))
    print('  - Total val non-occlu:', len(non_occlu_val_loader))
    print('  - Total val occlusion:', len(occlu_val_loader))
    num_batches = min(len(non_occlu_train_loader), len(occlu_train_loader))

    ''' Optimizer & Scheduler '''
    params_gen = []
    for name, p in model_generator.named_parameters():
        params_gen.append(p)

    print("[INFO] number trained params gen: ", len(params_gen))
    params_disc = [p for name, p in model_disciminator.named_parameters()]
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
