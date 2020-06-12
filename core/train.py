import numpy as np
import torch
from tools import MultiItemAverageMeter, accuracy


def train_an_epoch(config, loaders, base, current_epoch, train_gan, train_reid, train_pixel, optimize_sl_enc):

    # set train mode
    base.set_train()
    base.lr_scheduler_step(current_epoch)
    meter = MultiItemAverageMeter()


    # train loop
    for _ in range(200):

        # zero grad
        base.sl_enc_optimizer.zero_grad()
        base.gen_optimizer.zero_grad()
        base.dis_optimizer.zero_grad()
        base.il_enc_optimizer.zero_grad()

        #
        results = {}

        # gan
        if train_gan:
            gen_loss_without_feature, gen_loss_gan_feature, dis_loss, image_list = train_gan_an_iter(config, loaders, base)

            gen_loss_gan_feature.backward(retain_graph=True)
            base.sl_enc_optimizer.zero_grad()
            gen_loss_without_feature.backward()

            base.dis_optimizer.zero_grad()
            dis_loss.backward()

            results['gen_loss_gan_feature'] = gen_loss_gan_feature.data
            results['gen_loss_without_feature'] = gen_loss_without_feature.data
            results['dis_loss'] = dis_loss.data

        # pixel
        if train_pixel:
            assert train_gan
            pixel_loss = train_pixel_an_iter(config, loaders, base, image_list)
            (config.weight_pixel_loss*pixel_loss).backward()
            results['pixel_loss'] = pixel_loss.data

        # reid
        if train_reid:
            cls_loss, triplet_loss, acc = train_reid_an_iter(config, loaders, base)
            reid_loss = cls_loss + triplet_loss
            reid_loss.backward()
            results['cls_loss'] = cls_loss.data
            results['triplet_loss'] = triplet_loss.data
            results['acc'] = acc

        # optimize
        if optimize_sl_enc:
            base.sl_enc_optimizer.step()
        if train_gan:
            base.gen_optimizer.step()
            base.dis_optimizer.step()
        if train_reid:
            base.il_enc_optimizer.step()

        # record
        meter.update(results)

    return meter.get_str()


def train_gan_an_iter(config, loaders, base):

    ### load data
    rgb_images, rgb_ids, rgb_cams, _ = loaders.gan_rgb_train_iter.next_one()
    ir_images, ir_ids, ir_cams, _ = loaders.gan_ir_train_iter.next_one()
    rgb_images, ir_images = rgb_images.to(base.device), ir_images.to(base.device)
    rgb_ids, ir_ids = rgb_ids.to(base.device), ir_ids.to(base.device)
    assert torch.equal(rgb_ids, ir_ids)

    # encode
    real_rgb_contents, real_rgb_styles, real_rgb_predict = base.generator_rgb.module.encode(rgb_images, True)
    real_ir_contents, real_ir_styles, real_ir_predict = base.generator_ir.module.encode(ir_images, True)

    # decode (within domain)
    reconst_rgb_images = base.generator_rgb.module.decode(real_rgb_contents, real_rgb_styles)
    reconst_ir_images = base.generator_ir.module.decode(real_ir_contents, real_ir_styles)

    # decode (cross domain)
    fake_rgb_images = base.generator_rgb.module.decode(real_ir_contents, real_rgb_styles)
    fake_ir_images = base.generator_ir.module.decode(real_rgb_contents, real_ir_styles)

    # encode again
    fake_ir_contents, fake_rgb_styles, _ = base.generator_rgb.module.encode(fake_rgb_images, True)
    fake_rgb_contents, fake_ir_styles, _ = base.generator_ir.module.encode(fake_ir_images, True)

    # decode again
    cycreconst_rgb_images = base.generator_rgb.module.decode(fake_rgb_contents, real_rgb_styles)
    cycreconst_ir_images = base.generator_ir.module.decode(fake_ir_contents, real_ir_styles)


    ### generator

    # reconstruction loss
    gen_loss_reconst_images = base.reconst_loss(reconst_rgb_images, rgb_images) + base.reconst_loss(reconst_ir_images, ir_images)
    gen_loss_cyclereconst_images = base.reconst_loss(cycreconst_rgb_images, rgb_images) + base.reconst_loss(cycreconst_ir_images, ir_images)
    gen_loss_reconst_contents = base.reconst_loss(fake_rgb_contents, real_rgb_contents) + base.reconst_loss(fake_ir_contents, real_ir_contents)
    gen_loss_reconst_styles = base.reconst_loss(fake_rgb_styles, real_rgb_styles) + base.reconst_loss(fake_ir_styles, real_ir_styles)

    # gan loss
    gen_loss_gan = base.discriminator_rgb.module.calc_gen_loss(fake_rgb_images) + base.discriminator_ir.module.calc_gen_loss(fake_ir_images)

    # overall loss
    gen_loss_without_gan_feature = 1.0 * gen_loss_gan + \
                               base.config.weight_gan_image * (gen_loss_reconst_images + gen_loss_cyclereconst_images)
    gen_loss_gan_feature = base.config.weight_gan_feature * (gen_loss_reconst_contents + gen_loss_reconst_styles)

    # images list
    image_list = [rgb_images, fake_ir_images.detach(), ir_images, fake_rgb_images.detach()]

    ### discriminator
    dis_loss = base.discriminator_rgb.module.calc_dis_loss(fake_rgb_images.detach(), rgb_images) + \
               base.discriminator_ir.module.calc_dis_loss(fake_ir_images.detach(), ir_images)

    return gen_loss_without_gan_feature, gen_loss_gan_feature, dis_loss, image_list




def train_pixel_an_iter(config, loaders, base, image_list):


    rgb_images, fake_ir_images, ir_images, fake_rgb_images = image_list

    ### compute feature
    _, _, rgb_cls_score = base.encoder(rgb_images, True, sl_enc=False)
    _, _, fake_ir_score = base.encoder(fake_ir_images, True, sl_enc=False)
    _, _, ir_cls_score = base.encoder(ir_images, True, sl_enc=False)
    _, _, fake_rgb_score = base.encoder(fake_rgb_images, True, sl_enc=False)

    ###
    loss_rgb = base.kl_loss(fake_ir_score, rgb_cls_score)
    loss_ir = base.kl_loss(fake_rgb_score, ir_cls_score)

    ###
    return loss_rgb + loss_ir


def train_reid_an_iter(config, loaders, base):

    ### load data
    rgb_images, rgb_ids, rgb_cams, _ = loaders.reid_rgb_train_iter.next_one()
    ir_images, ir_ids, ir_cams, _ = loaders.reid_ir_train_iter.next_one()
    rgb_images, ir_images = rgb_images.to(base.device), ir_images.to(base.device)
    rgb_ids, ir_ids = rgb_ids.to(base.device), ir_ids.to(base.device)
    assert torch.equal(rgb_ids, ir_ids)

    ### compute feature
    _, rgb_feature_vectors, rgb_cls_score = base.encoder(rgb_images, True, sl_enc=False)
    _, ir_feature_vectors, ir_cls_score = base.encoder(ir_images, True, sl_enc=False)

    ### compute loss
    rgb_cls_loss = base.ide_creiteron(rgb_cls_score, rgb_ids)
    ir_cls_loss = base.ide_creiteron(ir_cls_score, ir_ids)
    cls_loss = (rgb_cls_loss + ir_cls_loss) / 2.0

    triplet_loss_1 = base.triplet_creiteron(rgb_feature_vectors, ir_feature_vectors, ir_feature_vectors, rgb_ids, ir_ids, ir_ids)
    triplet_loss_2 = base.triplet_creiteron(ir_feature_vectors, rgb_feature_vectors, rgb_feature_vectors, ir_ids, rgb_ids, rgb_ids)
    triplet_loss = (triplet_loss_1 + triplet_loss_2) / 2.0

    ### acc
    rgb_acc = accuracy(rgb_cls_score, rgb_ids, [1])[0]
    ir_acc = accuracy(ir_cls_score, ir_ids, [1])[0]
    acc = torch.Tensor([rgb_acc, ir_acc])

    return cls_loss, triplet_loss, acc

