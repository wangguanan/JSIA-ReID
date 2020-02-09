import os
import torch
from torchvision.utils import save_image
import numpy as np
from tools import make_dirs


def visualize(config, loaders, base, current_epoch):

    # set eval mode
    base.set_eval()

    # generate images
    with torch.no_grad():
        # fixed images
        rgb_images, ir_images = base.rgb_images, base.ir_images
        # encode
        real_rgb_contents, real_rgb_styles, _ = base.generator_rgb.module.encode(rgb_images, True)
        real_ir_contents, real_ir_styles, _ = base.generator_ir.module.encode(ir_images, True)
        # decode (cross domain)
        fake_rgb_images = base.generator_rgb.module.decode(real_ir_contents, real_rgb_styles)
        fake_ir_images = base.generator_ir.module.decode(real_rgb_contents, real_ir_styles)
        # encode again
        fake_ir_contents, fake_rgb_styles, _ = base.generator_rgb.module.encode(fake_rgb_images, True)
        fake_rgb_contents, fake_ir_styles, _ = base.generator_ir.module.encode(fake_ir_images, True)

        # # decode again
        # cycreconst_rgb_images = base.generator_rgb.module.decode(fake_rgb_contents, real_rgb_styles)
        # cycreconst_ir_images = base.generator_ir.module.decode(fake_ir_contents, real_ir_styles)
        #
        # cycreconst2_rgb_images = base.generator_rgb.module.decode(real_rgb_contents, fake_rgb_styles)
        # cycreconst2_ir_images = base.generator_ir.module.decode(real_ir_contents, fake_ir_styles)
        #
        # wrong_fake_rgb_style, shuffled_fake_rgb_style = shuffle_styles(fake_rgb_styles, config.gan_k)
        # wrong_fake_ir_style, shuffled_fake_ir_style = shuffle_styles(fake_ir_styles, config.gan_k)
        #
        # cycreconst3_rgb_images = base.generator_rgb.module.decode(fake_rgb_contents, shuffled_fake_rgb_style)
        # cycreconst3_ir_images = base.generator_ir.module.decode(fake_ir_contents, shuffled_fake_ir_style)
        #
        # cycreconst4_rgb_images = base.generator_rgb.module.decode(fake_rgb_contents, wrong_fake_rgb_style)
        # cycreconst4_ir_images = base.generator_ir.module.decode(fake_ir_contents, wrong_fake_ir_style)

    # save images
    images = (torch.cat([rgb_images, ir_images, fake_rgb_images, fake_ir_images,
                         # cycreconst_rgb_images, cycreconst_ir_images,
                         # cycreconst2_rgb_images, cycreconst2_ir_images,
                         # cycreconst3_rgb_images, cycreconst3_ir_images,
                         # cycreconst4_rgb_images, cycreconst4_ir_images
                         ], dim=0) + 1.0) / 2.0
    save_image(images.data.cpu(), os.path.join(config.save_images_path, '{}.jpg'.format(current_epoch)), config.gan_p*config.gan_k)



def shuffle_styles(styles, k):
    wrong_styles = torch.cat((styles[k:], styles[:k]), dim=0)

    random_index = []
    for i in range(styles.shape[0] // k):
        random_index.extend(list(np.random.permutation(k) + i * k))
    random_styles = styles[random_index]

    return wrong_styles, random_styles