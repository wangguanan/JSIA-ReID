import argparse
import os
import ast
import copy
import numpy as np

from core import Loaders, Base, train_an_epoch, test, visualize
from tools import Logger, make_dirs, time_now, os_walk


def main(config):

    # loaders and base
    loaders = Loaders(config)
    base = Base(config, loaders)

    # make dirs
    make_dirs(config.save_images_path)
    make_dirs(config.save_models_path)
    make_dirs(config.save_features_path)

    # logger
    logger = Logger(os.path.join(config.output_path, 'log.txt'))
    logger(config)


    if config.mode == 'train':

        # automatically resume model from the latest one
        start_train_epoch = 0
        root, _, files = os_walk(config.save_models_path)
        if len(files) > 0:
            # get indexes of saved models
            indexes = []
            for file in files:
                indexes.append(int(file.replace('.pkl', '').split('_')[-1]))

            # remove the bad-case and get available indexes
            model_num = len(base.model_list)
            available_indexes = copy.deepcopy(indexes)
            for element in indexes:
                if indexes.count(element) < model_num:
                    available_indexes.remove(element)

            available_indexes = sorted(list(set(available_indexes)), reverse=True)
            unavailable_indexes = list(set(indexes).difference(set(available_indexes)))

            if len(available_indexes) > 0: # resume model from the latest model
                base.resume_model(available_indexes[0])
                start_train_epoch = available_indexes[0] + 1
                logger('Time: {}, automatically resume training from the latest step (model {})'.format(time_now(), available_indexes[0]))
            else: #
                logger('Time: {}, there are no available models')


        # main loop
        for current_epoch in range(start_train_epoch, config.warmup_reid_epoches + config.warmup_gan_epoches + config.train_epoches):

            # train
            if current_epoch < config.warmup_reid_epoches: # warmup reid model
                results = train_an_epoch(config, loaders, base, current_epoch, train_gan=True, train_reid=True, train_pixel=False, optimize_sl_enc=True)
            elif current_epoch < config.warmup_reid_epoches + config.warmup_gan_epoches: # warmup GAN model
                results = train_an_epoch(config, loaders, base, current_epoch, train_gan=True, train_reid=False, train_pixel=False, optimize_sl_enc=False)
            else: # joint train
                results = train_an_epoch(config, loaders, base, current_epoch, train_gan=True, train_reid=True, train_pixel=True, optimize_sl_enc=True)
            logger('Time: {};  Epoch: {};  {}'.format(time_now(), current_epoch, results))

            # save model
            base.save_model(current_epoch)

        # test
        visualize(config, loaders, base, current_epoch)
        results = test(config, base, loaders, brief=False)
        for key in results.keys():
            logger('Time: {}\n Setting: {}\n {}'.format(time_now(), key, results[key]))


    elif config.mode == 'test':
        # resume from pre-trained model and test
        base.resume_model_from_path(config.pretrained_model_path, config.pretrained_model_epoch)
        results = test(config, base, loaders, brief=False)
        for key in results.keys():
            logger('Time: {}\n Setting: {}\n {}'.format(time_now(), key, results[key]))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='train')

    # dataset configuration
    parser.add_argument('--dataset_path', type=str, default='SYSU-MM01/')
    parser.add_argument('--image_size_4reid', type=int, nargs='+', default=[256, 128], help='image size when training reid')
    parser.add_argument('--image_size_4gan', type=int, nargs='+', default=[128, 64], help='image size when training gan. for saving memory, we use small size')
    parser.add_argument('--reid_p', type=int, default=16, help='person count in a batch')
    parser.add_argument('--reid_k', type=int, default=4, help='images count of a person in a batch')
    parser.add_argument('--gan_p', type=int, default=3, help='person count in a batch')
    parser.add_argument('--gan_k', type=int, default=3, help='images count of a person in a batch')

    # loss configuration
    parser.add_argument('--learning_rate_reid', type=float, default=0.00045)
    parser.add_argument('--weight_pixel_loss', type=float, default=0.01)
    parser.add_argument('--weight_gan_image', type=float, default=10.0)
    parser.add_argument('--weight_gan_feature', type=float, default=1.0)

    # train configuration
    parser.add_argument('--warmup_reid_epoches', type=int, default=0)
    parser.add_argument('--warmup_gan_epoches', type=int, default=600, help='our model is robust to this parameter, works well when larger than 100')
    parser.add_argument('--train_epoches', type=int, default=50)
    parser.add_argument('--milestones', type=int, nargs='+', default=[30])

    # logger configuration
    parser.add_argument('--output_path', type=str, default='out/base/')
    parser.add_argument('--max_save_model_num', type=int, default=2, help='0 for max num is infinit')
    parser.add_argument('--save_images_path', type=str, default=parser.parse_args().output_path+'images/')
    parser.add_argument('--save_models_path', type=str, default=parser.parse_args().output_path+'models/')
    parser.add_argument('--save_features_path', type=str, default=parser.parse_args().output_path+'features/')

    # test configuration
    parser.add_argument('--modes', type=str, nargs='+', default=['all', 'indoor'], help='indoor, all')
    parser.add_argument('--number_shots', type=str, nargs='+', default=['single', 'multi'], help='single, multi')
    parser.add_argument('--pretrained_model_path', type=str, default='')
    parser.add_argument('--pretrained_model_epoch', type=str, default='')

    # run
    config = parser.parse_args()
    config.milestones = list(np.array(config.milestones) + config.warmup_reid_epoches + config.warmup_gan_epoches)

    main(config)









