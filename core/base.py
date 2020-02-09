import sys
sys.path.append('..')
import os
import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from .networks import Generator, Discriminator, Encoder, weights_init
from tools import os_walk, time_now, make_dirs, CrossEntropyLabelSmooth, TripletLoss


class Base:

    def __init__(self, config, loaders):

        self.config=config
        self.loaders=loaders
        self.device=torch.device('cuda')

        self._init_networks()
        self._init_optimizers()
        self._init_creterion()
        self._init_fixed_items()


    def _init_networks(self):


        # init models
        self.encoder = Encoder(395) # include set-level(sl) and instance-level(sl) encoders
        self.generator_rgb = Generator(self.encoder) # include modality-specific encoder and decoder for RGB images
        self.generator_ir = Generator(self.encoder) # include modality-specific encoder and decoder for IR images
        '''as shown above, the two generators shares the same sel-level(sl) encoder'''
        self.discriminator_rgb = Discriminator(n_layer=4, middle_dim=32, num_scales=2)
        self.discriminator_ir = Discriminator(n_layer=4, middle_dim=32, num_scales=2)

        self.discriminator_rgb.apply(weights_init('gaussian'))
        self.discriminator_ir.apply(weights_init('gaussian'))

        # data parallel
        self.generator_rgb = nn.DataParallel(self.generator_rgb).to(self.device)
        self.generator_ir = nn.DataParallel(self.generator_ir).to(self.device)
        self.discriminator_rgb = nn.DataParallel(self.discriminator_rgb).to(self.device)
        self.discriminator_ir = nn.DataParallel(self.discriminator_ir).to(self.device)

        # recored all models for saving and loading
        self.model_list = []
        self.model_list.append(self.generator_rgb)
        self.model_list.append(self.generator_ir)
        self.model_list.append(self.discriminator_rgb)
        self.model_list.append(self.discriminator_ir)



    def _init_optimizers(self):

        sl_enc_params = list(self.encoder.resnet_conv1.parameters())
        il_enc_params = list(self.encoder.resnet_conv2.parameters()) + list(self.encoder.classifier.parameters())
        gen_params = list(self.generator_rgb.parameters()) + list(self.generator_ir.parameters())
        gen_params = list(set(gen_params).difference(set(sl_enc_params).union(set(il_enc_params))))
        dis_params = list(self.discriminator_rgb.parameters()) + list(self.discriminator_ir.parameters())

        sl_enc_params = [p for p in sl_enc_params if p.requires_grad]
        gen_params = [p for p in gen_params if p.requires_grad]
        dis_params = [p for p in dis_params if p.requires_grad]
        il_enc_params = [p for p in il_enc_params if p.requires_grad]

        self.sl_enc_optimizer = optim.Adam(sl_enc_params, lr=self.config.learning_rate_reid, betas=[0.9, 0.999], weight_decay=5e-4)
        self.gen_optimizer = optim.Adam(gen_params, lr=0.0001, betas=[0.5, 0.999], weight_decay=0.0001)
        self.dis_optimizer = optim.Adam(dis_params, lr=0.0001, betas=[0.5, 0.999], weight_decay=0.0001)
        self.il_enc_optimizer = optim.Adam(il_enc_params, lr=self.config.learning_rate_reid, betas=[0.9, 0.999], weight_decay=5e-4)

        self.sl_enc_lr_scheduler = optim.lr_scheduler.MultiStepLR(self.sl_enc_optimizer, milestones=self.config.milestones, gamma=0.1)
        self.gen_lr_scheduler = optim.lr_scheduler.MultiStepLR(self.gen_optimizer, milestones=self.config.milestones, gamma=0.1)
        self.dis_lr_scheduler = optim.lr_scheduler.MultiStepLR(self.dis_optimizer, milestones=self.config.milestones, gamma=0.1)
        self.il_enc_lr_scheduler = optim.lr_scheduler.MultiStepLR(self.il_enc_optimizer, milestones=self.config.milestones, gamma=0.1)


    def lr_scheduler_step(self, current_epoch):
        self.sl_enc_lr_scheduler.step(current_epoch)
        self.gen_lr_scheduler.step(current_epoch)
        self.dis_lr_scheduler.step(current_epoch)
        self.il_enc_lr_scheduler.step(current_epoch)


    def _init_creterion(self):
        self.reconst_loss = nn.L1Loss()
        self.id_loss = nn.CrossEntropyLoss()
        self.ide_creiteron = CrossEntropyLabelSmooth(395)
        self.triplet_creiteron = TripletLoss(0.3, 'euclidean')


    def kl_loss(self, score1, score2,  mini=1e-8):
        score2 = score2.detach()
        prob1 = F.softmax(score1, dim=1)
        prob2 = F.softmax(score2, dim=1)
        loss = torch.sum(prob2 * torch.log(mini + prob2 / (prob1 + mini)), 1) + \
                 torch.sum(prob1 * torch.log(mini + prob1 / (prob2 + mini)), 1)
        return loss.mean()


    def _init_fixed_items(self):

        rgb_images, _, _, _ = self.loaders.gan_rgb_train_iter.next_one()
        ir_images, _, _, _ = self.loaders.gan_ir_train_iter.next_one()
        self.rgb_images = rgb_images.to(self.device)
        self.ir_images = ir_images.to(self.device)


    def save_model(self, save_epoch):

        # save model
        for ii, _ in enumerate(self.model_list):
            torch.save(self.model_list[ii].state_dict(), os.path.join(self.config.save_models_path, 'model-{}_{}.pkl'.format(ii, save_epoch)))

        # if saved model is more than max num, delete the model with smallest epoch
        if self.config.max_save_model_num > 0:
            root, _, files = os_walk(self.config.save_models_path)

            # get indexes of saved models
            indexes = []
            for file in files:
                indexes.append(int(file.replace('.pkl', '').split('_')[-1]))

            # remove the bad-case and get available indexes
            model_num = len(self.model_list)
            available_indexes = copy.deepcopy(indexes)
            for element in indexes:
                if indexes.count(element) < model_num:
                    available_indexes.remove(element)

            available_indexes = sorted(list(set(available_indexes)), reverse=True)
            unavailable_indexes = list(set(indexes).difference(set(available_indexes)))

            # delete all unavailable models
            for unavailable_index in unavailable_indexes:
                try:
                    # os.system('find . -name "{}*_{}.pkl" | xargs rm  -rf'.format(self.config.save_models_path, unavailable_index))
                    for ii in range(len(self.model_list)):
                        os.remove(os.path.join(root, 'model-{}_{}.pkl'.format(ii, unavailable_index)))
                except:
                    pass

            # delete extra models
            if len(available_indexes) >= self.config.max_save_model_num:
                for extra_available_index in available_indexes[self.config.max_save_model_num:]:
                    # os.system('find . -name "{}*_{}.pkl" | xargs rm  -rf'.format(self.config.save_models_path, extra_available_index))
                    for ii in range(len(self.model_list)):
                        os.remove(os.path.join(root, 'model-{}_{}.pkl'.format(ii, extra_available_index)))


    def resume_model(self, resume_epoch):
        for ii, _ in enumerate(self.model_list):
            self.model_list[ii].load_state_dict(
                torch.load(os.path.join(self.config.save_models_path, 'model-{}_{}.pkl'.format(ii, resume_epoch))))
        print('Time: {}, successfully resume model from {}'.format(time_now(), resume_epoch))


    ## resume model from a path
    def resume_model_from_path(self, path, resume_epoch):
        for ii, _ in enumerate(self.model_list):
            self.model_list[ii].load_state_dict(
                torch.load(os.path.join(path, 'model-{}_{}.pkl'.format(ii, resume_epoch))))
        print('Time: {}, successfully resume model from {}'.format(time_now(), resume_epoch))


    ## set model as train mode
    def set_train(self):
        for ii, _ in enumerate(self.model_list):
            self.model_list[ii] = self.model_list[ii].train()


    ## set model as eval mode
    def set_eval(self):
        for ii, _ in enumerate(self.model_list):
            self.model_list[ii] = self.model_list[ii].eval()
