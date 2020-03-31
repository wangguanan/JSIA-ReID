import torch
import random
import math


class RandomErasing(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=(0.4914, 0.4822, 0.4465)):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):

        if random.uniform(0, 1) >= self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img

        return img


def rgb2gray(imgs, de_norm, norm, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], cuda=True):
    '''
    Params:
        imgs: torch.tensor, [bs, c, h, w], c=rgb
    '''

    device = torch.device('cuda') if cuda else torch.device('cpu')

    new_imgs = imgs.clone()

    if de_norm: # denormalize to [0, 1] with mean and std
        mean = torch.tensor(mean).view([1, 3, 1, 1]).repeat([new_imgs.size(0), 1, new_imgs.size(2), new_imgs.size(3)]).to(device)
        std = torch.tensor(std).view([1, 3, 1, 1]).repeat([new_imgs.size(0), 1, new_imgs.size(2), new_imgs.size(3)]).to(device)
        new_imgs = new_imgs * std + mean

    # to gray scale
    weights = torch.tensor([0.2989, 0.5870, 0.1140])
    weights = weights.view([1, 3, 1, 1]).repeat([new_imgs.shape[0], 1, new_imgs.shape[2], new_imgs.shape[3]]).to(device)
    new_imgs = (new_imgs * weights).sum(dim=1, keepdim=True).repeat([1, 3, 1, 1])

    if norm: # normalize with mean and std
        new_imgs = (new_imgs - mean) / std

    return new_imgs



def norm(x, mean, std, device=torch.device('cuda')):
    mean = torch.tensor(mean).view([1, 3, 1, 1]).repeat([x.size(0), 1, x.size(2), x.size(3)]).to(device)
    std = torch.tensor(std).view([1, 3, 1, 1]).repeat([x.size(0), 1, x.size(2), x.size(3)]).to(device)
    x = (x - mean) / std
    return x


def denorm(x, mean, std, device=torch.device('cuda')):
    mean = torch.tensor(mean).view([1, 3, 1, 1]).repeat([x.size(0), 1, x.size(2), x.size(3)]).to(device)
    std = torch.tensor(std).view([1, 3, 1, 1]).repeat([x.size(0), 1, x.size(2), x.size(3)]).to(device)
    x = x * std + mean
    return x



