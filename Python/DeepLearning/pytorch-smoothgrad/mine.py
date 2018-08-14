# coding=utf-8
import argparse
import os
import sys

import numpy as np
from scipy import misc
import cv2
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.models import vgg16, vgg19, resnet50, alexnet, resnet18, inception_v3
from torchvision.utils import save_image

from lib.gradients import VanillaGrad, SmoothGrad, GuidedBackpropGrad, GuidedBackpropSmoothGrad
from lib.image_utils import preprocess_image, save_as_gray_image, save_as_rgb_image
from lib.labels import IMAGENET_LABELS

import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.cuda = False
    args.img = './dog.jpg'
    args.out_dir = './result/grad/'
    args.n_samples = 50

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    target_layer_names = ['35']
    target_index = None

    # Prepare input image
    if args.img:
        img = cv2.imread(args.img, 1)
    else:
        img = misc.face()
    img = np.float32(cv2.resize(img, (224, 224))) / 255
    preprocessed_img = preprocess_image(img, args.cuda)

    model = vgg16(pretrained=True)
    if args.cuda:
        model.cuda()

    # Prediction
    output = model(preprocessed_img)
    pred_index = np.argmax(output.data.cpu().numpy())
    print('Prediction: {}'.format(IMAGENET_LABELS[pred_index]))

    # Compute vanilla gradient
    vanilla_grad = VanillaGrad(
        pretrained_model=model, cuda=args.cuda)
    vanilla_saliency = vanilla_grad(preprocessed_img, index=target_index)
    save_as_gray_image(vanilla_saliency, os.path.join(args.out_dir, 'vanilla_grad.jpg'))
    print('Saved vanilla gradient image')

    # Reload preprocessed image
    preprocessed_img = preprocess_image(img, args.cuda)

    # Compute guided gradient
    guided_grad = GuidedBackpropGrad(
        pretrained_model=model, cuda=args.cuda)
    guided_saliency = guided_grad(preprocessed_img, index=target_index)
    save_as_gray_image(guided_saliency, os.path.join(args.out_dir, 'guided_grad.jpg'))
    print('Saved guided backprop gradient image')

    # Reload preprocessed image
    preprocessed_img = preprocess_image(img, args.cuda)

    # Compute smooth gradient
    smooth_grad = SmoothGrad(
        pretrained_model=model,
        cuda=args.cuda,
        n_samples=args.n_samples,
        magnitude=True)
    smooth_saliency = smooth_grad(preprocessed_img, index=target_index)
    save_as_gray_image(smooth_saliency, os.path.join(args.out_dir, 'smooth_grad.jpg'))
    print('Saved smooth gradient image')

    # Reload preprocessed image
    preprocessed_img = preprocess_image(img, args.cuda)

    # Compute guided smooth gradient
    guided_smooth_grad = GuidedBackpropSmoothGrad(
        pretrained_model=model,
        cuda=args.cuda,
        n_samples=args.n_samples,
        magnitude=True)
    guided_smooth_saliency = guided_smooth_grad(preprocessed_img, index=target_index)
    save_as_gray_image(guided_smooth_saliency, os.path.join(args.out_dir, 'guided_smooth_grad.jpg'))
    print('Saved guided backprop smooth gradient image')


def noise_add():
    img_path = './dog.jpg'
    img = cv2.imread(img_path, 1)
    img = img[..., ::-1]
    img = np.float32(img) / 255
    noise = np.random.normal(0, 0.01, img.shape).astype(np.float32)
    print np.max(img) - np.min(img)
    out = (img + 0.85*noise)
    assert(out.all != img.all)
    plt.subplot(131)
    plt.imshow(img)
    plt.axis('off')
    plt.subplot(132)
    plt.imshow(noise)
    plt.axis('off')
    plt.subplot(133)
    plt.imshow(out)
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    # main()
    noise_add()
