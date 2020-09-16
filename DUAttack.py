from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.models as models
import torch.utils.data.sampler as sp
from torchvision.utils import save_image

import os, sys
lib_path = os.path.abspath(os.path.join('..'))
sys.path.append(lib_path)
import argparse
import gc
import csv
import random
import numpy as np
import scipy.misc
import imageio
from scipy.io import savemat,loadmat

from sklearn.externals import joblib
from model.net import Net_ll
from model.vgg import VGG
from model.resnet import ResNet50
from utils.attack import Attack
from utils.qmnist import QMNIST

seed = 1000
torch.manual_seed(seed)            # for cpu
torch.cuda.manual_seed(seed)       # for current gpu
np.random.seed(seed)
random.seed(1000)


def tensor2im(im):
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    if not isinstance(im, np.ndarray):
        if isinstance(im, torch.Tensor):
            image_tensor = im.data
        else:
            return im
        im_np = image_tensor.cpu().float().numpy()
        for i in range(len(mean)):
            im_np[i] = im_np[i] * std[i] + mean[i]
        im_np = im_np * 255
        im_np = np.transpose(im_np, (1, 2, 0))  # (c, h, w) -> (h, w, c)
    else:
        im_np = im
    return im_np.astype(np.uint8)


def cal_azure(model, data):
    data = data.view(data.size(0), 784).cpu().numpy()
    output = model.predict(data)
    output = torch.from_numpy(output).cuda().long()
    return output


def pred(bs, imgs, ori_labels, tar_labels, net, flag=0):
    tar_vec = torch.ones(bs)
    untar_vec = torch.ones(bs)
    if flag == 0:
        with torch.no_grad():
            out = net(imgs)
        _, pred = torch.max(out.data, 1)
    else:
        with torch.no_grad():
            pred = cal_azure(net, imgs)
    untar_vec = pred.eq(ori_labels)
    tar_vec = pred.ne(tar_labels)
    # delete cache
    torch.cuda.empty_cache()
    gc.collect()
    return untar_vec, tar_vec


def compute_dist(bs, vec_adv, vec_ori, lf_norm):
    final_vec = torch.zeros(bs)
    lf_tmp = torch.zeros(bs)
    correct = 0.
    total = 0.
    final_vec = torch.add(vec_adv.float(), vec_ori.float())
    correct = (final_vec == 2).sum().float() # classify correctly after adding UAP on imngs classified correctly originally
    total = (vec_ori == 1).sum()  # classify correctly originally
    if correct == total:
        lf_dist = 0.0
    else:
        lf_ori_correct = lf_norm.cpu() * vec_ori.cpu().float()
        lf_adv_correct = lf_norm.cpu() * (final_vec == 2).cpu().float()
        lf_dist = ((lf_ori_correct - lf_adv_correct).sum()) / (
                   (lf_ori_correct - lf_adv_correct).ne(lf_tmp).sum().float())
    return correct, total, lf_dist


def find_unique_class(samples, labels, n=500):
    data_n = torch.zeros_like(samples).cuda()
    label_n = torch.zeros_like(labels).cuda()
    for i in range(num_classes):
        for j in range(1000):
            temp = (label_n - labels[j]).abs()
            if (temp == 0).nonzero().numel() == 0:
                data_n[i] = samples[j]
                label_n[i] = labels[j]
    print(label_n[0:n])
    return data_n[0:n], label_n[0:n]


def parse_args():

    parser = argparse.ArgumentParser(description='DUAttack')
    parser.add_argument('--root', dest='root',
                        help='where dataset exit',
                        default='/datassd/Dataset', type=str)
    parser.add_argument('--dir', dest='record_dir',
                        help='directory for recording the results',
                        default='./records/results_duattack.csv', type=str)
    parser.add_argument('--model', dest='model',
                        help='dir for loading the weights',
                        default='./weights', type=str)
    parser.add_argument('--data', dest='dataset',
                         help='mnist, cifar10, imagenet',
                         default='imagenet', type=str)
    parser.add_argument('--nw', dest='num_works',
                        help='number of worker to load data',
                        default=4, type=int)
    parser.add_argument('--bs', dest='batch_size',
                        default=500, type=int)
    parser.add_argument('--trainlist', dest='trainlist',
                        help='choose how many data for computing the UAP',
                        default=500, type=int)
    parser.add_argument('--testlist', dest='testlist',
                        help='choose how many data for validating the performance of UAP',
                        default=10000, type=int)
    parser.add_argument('--tar', dest='target',
                        help='False for untarget, True for target',
                        default=False, type=bool)
    parser.add_argument('--tar_cls', dest='tar_cls',
                        help='the targeted label when True for target',
                        default=0, type=int)
    # ----------------------------[Params for DUAttack]---------------------------------------
    parser.add_argument('--method', dest='method',
                        help='update methods: random, eyematrix',
                        default='eyematrix', type=str)
    parser.add_argument('--iter', dest='iter',
                        help='1000 for imagenet and cifar10, 500 for mnist',
                        default=1000, type=int)
    parser.add_argument('--eps', dest='eps',
                        help='0.2 for imagenet and cifar10, 0.02 for mnist',
                        default=0.2, type=float)
    parser.add_argument('--dist', dest='dist',
                        help='70.0 and 20.0 for imagenet, 16.0 and 4.0 for cifar10, 4.95 for mnist',
                        default=70.0, type=float)
    # --------------------------------------------------------------------------------------
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parse_args()
    print('Called with args:')
    print(args)
    flag_mnist = 0

    # Dataset
    valdir = os.path.join(args.root, 'ImageNet/val')
    std=[0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    trainset_list = [i for i in range(args.trainlist)]
    testset_list = [(49999 - i) for i in range(args.testlist)]
    if args.dataset == 'imagenet':
        trainloader = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(valdir,
                                                  transforms.Compose([
                                                      transforms.Resize(256),
                                                      transforms.CenterCrop(224),
                                                      transforms.ToTensor(),
                                                      normalize,
                                                  ])),
                                                  batch_size=1000,
                                                  shuffle=True,
                                                  num_workers=args.num_works,
                                                  pin_memory=True)
        testloader = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(valdir,
                                                 transforms.Compose([
                                                     transforms.Resize(256),
                                                     transforms.CenterCrop(224),
                                                     transforms.ToTensor(),
                                                     normalize,
                                                 ])),
                                                 batch_size=args.batch_size,
                                                 sampler=sp.SubsetRandomSampler(testset_list),
                                                 num_workers=args.num_works)
    elif args.dataset == 'cifar10':
        trainloader = torch.utils.data.DataLoader(torchvision.datasets.CIFAR10(
                                                  root=args.root,
                                                  train=True,
                                                  download=False,
                                                  transform=transforms.Compose([
                                                      transforms.ToTensor(),
                                                  ])),
                                                  batch_size=args.trainlist,
                                                  sampler=sp.SubsetRandomSampler(trainset_list),
                                                  num_workers=args.num_works,
                                                  pin_memory=True)
        testloader = torch.utils.data.DataLoader(torchvision.datasets.CIFAR10(
                                                  root=args.root,
                                                  train=False,
                                                  download=False,
                                                  transform=transforms.Compose([
                                                      transforms.ToTensor(),
                                                  ])),
                                                  batch_size=args.batch_size,
                                                  shuffle=False,
                                                  num_workers=args.num_works)
    elif args.dataset == 'mnist':
        trainloader = torch.utils.data.DataLoader(torchvision.datasets.MNIST(
                                                  root=args.root,
                                                  train=True,
                                                  download=True,
                                                  transform=transforms.Compose([
                                                      transforms.ToTensor(),
                                                  ])),
                                                  batch_size=args.batch_size,
                                                  sampler=sp.SubsetRandomSampler(trainset_list),
                                                  num_workers=args.num_works,
                                                  pin_memory=True)
        testloader = torch.utils.data.DataLoader(torchvision.datasets.MNIST(
                                                  root=args.root,
                                                  train=False,
                                                  download=True,
                                                  transform=transforms.Compose([
                                                      transforms.ToTensor(),
                                                  ])),
                                                  batch_size=args.batch_size,
                                                  shuffle=False,
                                                  num_workers=args.num_works)
    else:
        qtrain = QMNIST('_qmnist', train=True, compat=True, download=True, transform=transforms.ToTensor())
        trainloader = torch.utils.data.DataLoader(qtrain,
                                                  batch_size=args.batch_size,
                                                  sampler=sp.SubsetRandomSampler(trainset_list),
                                                  num_workers=args.num_works)
        qtest10k = QMNIST('_qmnist', what='test10k', compat=True, download='True', transform=transforms.ToTensor())
        testloader = torch.utils.data.DataLoader(qtest10k,
                                                batch_size=args.batch_size,
                                                shuffle=False,
                                                num_workers=args.num_works)

    # Net
    load_dir = args.model
    if args.dataset == 'imagenet':
        D1 = models.vgg16(pretrained=True).cuda()
        D1 = nn.DataParallel(D1)
        D1.eval()

        D2 = models.resnet50(pretrained=True).cuda()
        D2 = nn.DataParallel(D2)
        D2.eval()

        num_classes = 1000
        img_size = 224
        img_ch = 3
        clipmax = [2.249, 2.429, 2.640]
        clipmin = [-2.118, -2.036, -1.804]
    elif args.dataset == 'cifar10':
        D1_path = os.path.join(load_dir, 'D1_network.pth')
        D1 = VGG('VGG16').cuda()
        D1.load_state_dict(torch.load(D1_path))
        D1.eval()

        D2_path = os.path.join(load_dir, 'D2_network.pth')
        D2 = ResNet50().cuda()
        D2.load_state_dict(torch.load(D2_path))
        D2.eval()

        num_classes = 10
        img_size = 32
        img_ch = 3
        clipmax = 1.
        clipmin = 0.
    else:
        D2_path = os.path.join(load_dir, 'Netll_mnist.pth')
        D2= Net_ll().cuda()
        D2.load_state_dict(torch.load(D2_path))
        D2.eval()

        D1_path = os.path.join(load_dir, 'sklearn_mnist_model.pkl')
        D1 = joblib.load(D1_path)

        num_classes = 10
        img_size = 28
        img_ch = 1
        clipmax = 1.
        clipmin = 0.
        flag_mnist = 1

    # Criterion and if target
    criterion = nn.CrossEntropyLoss()
    if args.target:
        tar_cls = args.tar_cls
        print('perform target attack and tar_cls is ', tar_cls)
    else:
        tar_cls = -1
        print('perform untarget attack')

    # Compute the UAP
    Duattack = Attack(imgsize=img_size, net=D1,
                      clip_min=clipmin, clip_max=clipmax,
                      criterion=criterion, mu=args.eps)
    # init the zeros_img
    adv_mask_tmp = torch.zeros(1, img_ch, img_size, img_size)
    adv_mask_tmp = Variable(adv_mask_tmp).cuda()
    adv_mask = adv_mask_tmp
    print('computing the UAP by DUAttack')
    for i, data in enumerate(trainloader, 0):
        print('i:', i)
        if img_ch == 3:
            D1.eval()
        # original data
        ori_img, ori_label = data
        ori_img = Variable(ori_img).cuda()
        ori_label = Variable(ori_label).cuda()
        if args.dataset == 'imagenet':
            ori_imgs, ori_labels = find_unique_class(samples=ori_img, labels=ori_label, n=args.trainlist)
        else:
            ori_imgs = ori_img
            ori_labels = ori_label
        print('train data is ready!')
        bs = ori_imgs.size()[0]
        print('bs: ', bs)

        # target labels
        tar_labels = torch.ones(bs) * tar_cls
        tar_labels = Variable(tar_labels).long().cuda()
        if args.target:
            ori_labels = tar_labels

        # generate the UAP
        if args.method == 'eyematrix':
            adv_mask_tmp = Duattack.DD_label_m1(x=ori_imgs, mask=adv_mask,
                                                y=ori_labels,
                                                targeted=args.target,
                                                eps=args.eps,
                                                iteration=args.iter,
                                                dist=args.dist,
                                                rand=False)
        else:
            adv_mask_tmp = Duattack.DD_label_m1(x=ori_imgs, mask=adv_mask,
                                                y=ori_labels,
                                                targeted=args.target,
                                                eps=args.eps,
                                                iteration=args.iter,
                                                dist=args.dist,
                                                rand=True)


        adv_mask = adv_mask_tmp
        print('adv_mask: ', torch.norm(adv_mask).item())
        if i==0 and img_size==224:
            print('end for train')
            break

    # delete cache
    del bs, ori_imgs, ori_labels, tar_labels
    torch.cuda.empty_cache()
    gc.collect()

    # Validate the performance of UAP on testset
    D1_correct = 0.
    D1_total = 0.
    D2_correct = 0.
    D2_total = 0.

    D1_lf_dist = 0.
    D2_lf_dist = 0.     # Lf norm
    cnt = 0.
    d_cnt = 0.
    for i, data in enumerate(testloader, 0):
        D2.eval()
        if img_ch == 3:
            D1.eval()

        # original data
        ori_imgs, ori_labels = data
        ori_imgs = Variable(ori_imgs).cuda()
        ori_labels = Variable(ori_labels).cuda()

        # variables for recording
        bs = ori_imgs.size()[0]
        lf_norm = torch.zeros(bs)

        # target labels
        tar_labels = torch.ones(bs) * tar_cls
        tar_labels = Variable(tar_labels).long().cuda()
        if args.target:
            ori_labels = tar_labels

        # D1: original output
        D1_untar_ori, D1_tar_ori = pred(bs=bs, imgs=ori_imgs,
                                        ori_labels=ori_labels,
                                        tar_labels=tar_labels,
                                        net = D1, flag=flag_mnist)

        # D2: original output
        D2_untar_ori, D2_tar_ori = pred(bs=bs, imgs=ori_imgs,
                                        ori_labels=ori_labels,
                                        tar_labels=tar_labels,
                                        net = D2)

        # add the UAP with the clean imgs
        adv_imgs = ori_imgs + adv_mask
        if img_size == 224:
            for kk in range(img_ch):
                adv_imgs[:,kk,:,:] = torch.clamp(adv_imgs[:,kk,:,:], clipmin[kk], clipmax[kk])
        else:
            adv_imgs = torch.clamp(adv_imgs, clipmin, clipmax)
        # compute attack distance for this batch
        perturb = adv_imgs - ori_imgs
        if img_size == 224:
            for i in range(3):
                perturb[:, i, :, :] = perturb[:, i, :, :] * std[i] * 255.0
        else:
            pass
        lf_norm = perturb.view(bs, -1).norm(2, 1)
        cnt += adv_imgs.size()[0]

        # D1: adversarial output
        D1_untar_adv, D1_tar_adv = pred(bs=bs, imgs=adv_imgs,
                                        ori_labels=ori_labels,
                                        tar_labels=tar_labels,
                                        net = D1, flag=flag_mnist)
        # D2: adversarial output
        D2_untar_adv, D2_tar_adv = pred(bs=bs, imgs=adv_imgs,
                                        ori_labels=ori_labels,
                                        tar_labels=tar_labels,
                                        net = D2)

        # compute asr and distance
        d_cnt += 1
        if args.target is False:
            # untarget
            correct1, total1, lf_dist1 = compute_dist(bs=bs, vec_adv=D1_untar_adv,
                                                      vec_ori=D1_untar_ori, lf_norm=lf_norm)
            D1_correct += correct1
            D1_total += total1
            D1_lf_dist += lf_dist1

            correct2, total2, lf_dist2 = compute_dist(bs=bs, vec_adv=D2_untar_adv,
                                                      vec_ori=D2_untar_ori, lf_norm=lf_norm)
            D2_correct += correct2
            D2_total += total2
            D2_lf_dist += lf_dist2
        else:
            # target
            correct1, total1, lf_dist1 = compute_dist(bs=bs, vec_adv=D1_tar_adv,
                                                      vec_ori=D1_tar_ori, lf_norm=lf_norm)
            D1_correct += correct1
            D1_total += total1
            D1_lf_dist += lf_dist1

            correct2, total2, lf_dist2 = compute_dist(bs=bs, vec_adv=D2_tar_adv,
                                                      vec_ori=D2_tar_ori, lf_norm=lf_norm)
            D2_correct += correct2
            D2_total += total2
            D2_lf_dist += lf_dist2
        # delete cache
        torch.cuda.empty_cache()
        gc.collect()

    print('Total imgs for validation: ', cnt)
    D1_asr = 100. - 100. * D1_correct.float() / D1_total
    D2_asr = 100. - 100. * D2_correct.float() / D2_total
    print('DUAttack, ASR on D1 and D2: %.2f %%, %.2f %%' % (D1_asr, D2_asr))
