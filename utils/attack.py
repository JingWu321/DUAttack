import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import gc
import random
import collections


def myeyematrix(eye_mt, k, cols):
    tmp = np.eye(cols)
    for i in range(cols - k):
        tmp[:, i+k] = eye_mt[:, i]
    for j in range(k):
        tmp[:, j] = eye_mt[:, cols-k+j]
    return tmp


def cal_azure(model, data):
    data = data.view(data.size(0), 784).cpu().numpy()
    output = model.predict(data)
    output = torch.from_numpy(output).cuda().long()
    return output


def mean_square_distance(x1, x2, min_, max_):
    return np.mean((x1 - x2).cpu().numpy() ** 2) / ((max_ - min_) ** 2 + 1e-8)


class Attack(object):
    """
    DUAttack
    """
    def __init__(self, imgsize, net, clip_min, clip_max, mu=0.01, criterion=None):
        self.net = net
        self.criterion = criterion
        if self.criterion is None:
            self.criterion = nn.CrossEntropyLoss(reduction="sum")

        self.clipmin = clip_min
        self.clipmax = clip_max
        self.imgsize = imgsize
        self.mu = mu
        self.eye_ori = np.eye(self.imgsize)

    # label-based with momentum
    def DD_label_m1(self, x, mask, y=None, targeted=False,
                    eps=0.2, iteration=1000, dist=1., rand=False):

        # init some variables
        record_l = 0
        record_r = 0
        record = []
        mask_adv = mask
        x_adv = x
        bs = x.size()[0]
        img_ch = x.size()[1]
        flag = 0
        # recored the history
        momentum = torch.zeros(1, img_ch, self.imgsize, self.imgsize).cuda()
        # start
        for i in range(iteration):
            l_remaining = torch.zeros(bs)
            r_remaining = torch.zeros(bs)

            # 【1】select the c-th channel and the k-th eye matrix randomly
            # tmp_mask: the perturbation
            tmp = torch.zeros(1, img_ch, self.imgsize, self.imgsize)
            if flag == 0:
                c = torch.randint(0, img_ch, (1,))
                k = torch.randint(0, self.imgsize, (1,))
            if rand is False:
                eye_mask = myeyematrix(self.eye_ori, k, self.imgsize)
                tmp[:, c, :, :] = torch.tensor(eye_mask).float()
            else:
                # mask under the random selection
                random_mask = random.sample(range(0, self.imgsize*self.imgsize),
                                            self.imgsize*self.imgsize)
                random_mask = torch.tensor(random_mask)
                random_mask = random_mask.view(self.imgsize, self.imgsize)
                tmp[:, c, :, :] = random_mask.float()
            tmp_mask = Variable(tmp).cuda()

            # 【2.0】subtract
            left_mask = mask_adv - (eps + 0.9*momentum)*tmp_mask
            # keep the lf distance unchanged
            left_dist = torch.norm(left_mask)
            left_mask = (left_mask / (left_dist + 1e-8)) * dist
            # get the adversarial output
            left_adv = x + left_mask
            if self.imgsize == 224:
                for kk in range(img_ch):
                    left_adv[:,kk,:,:] = torch.clamp(left_adv[:,kk,:,:], self.clipmin[kk], self.clipmax[kk])
            else:
                left_adv = torch.clamp(left_adv, self.clipmin, self.clipmax)
            with torch.no_grad():
                if img_ch == 1:
                    left_preds = cal_azure(self.net, left_adv)
                else:
                    left_out = self.net(x=left_adv)
                    _, left_preds = torch.max(left_out.data, 1)
            # check
            if targeted:
                l_remaining = left_preds.ne(y)
            else:
                l_remaining = left_preds.eq(y)
            # if all images are misclassified, then break
            if l_remaining.sum() == 0:
                record.append((i,1))
                record_l += 1
                x_adv = left_adv
                mask_adv = (x_adv[0] - x[0]).unsqueeze(0)
                break

            torch.cuda.empty_cache()
            gc.collect()

            # 【2.1】add
            right_mask = mask_adv + (eps + 0.9*momentum)*tmp_mask
            # keep the lf distance unchanged
            right_dist = torch.norm(right_mask)
            right_mask = (right_mask / (right_dist + 1e-8)) * dist
            # get the adversarial output
            right_adv = x + right_mask
            if self.imgsize == 224:
                for kk in range(img_ch):
                    right_adv[:,kk,:,:] = torch.clamp(right_adv[:,kk,:,:], self.clipmin[kk], self.clipmax[kk])
            else:
                right_adv = torch.clamp(right_adv, self.clipmin, self.clipmax)
            with torch.no_grad():
                if img_ch == 1:
                    right_preds = cal_azure(self.net, right_adv)
                else:
                    right_out = self.net(x=right_adv)
                    _, right_preds = torch.max(right_out.data, 1)
            # check
            if targeted:
                r_remaining = right_preds.ne(y)
            else:
                r_remaining = right_preds.eq(y)
            # if all images are misclassified, then break
            if r_remaining.sum() == 0:
                record.append((i,2))
                record_r += 1
                x_adv = right_adv
                mask_adv = (x_adv[0] - x[0]).unsqueeze(0)
                break

            # 【3】compare
            # if add has less improve than subtract
            if r_remaining.sum() > l_remaining.sum():
                record.append((i,1))
                record_l += 1
                x_adv = left_adv
                mask_adv = (x_adv[0] - x[0]).unsqueeze(0)
                momentum -= tmp_mask*eps
            else:
                record.append((i,2))
                record_r += 1
                x_adv = right_adv
                mask_adv = (x_adv[0] - x[0]).unsqueeze(0)
                momentum += tmp_mask*eps
            flag = 0

            print('DD_label_Iter %d: %.2f' % (i, torch.norm(mask_adv)))
            print('record_r and record_l: %d, %d' % (record_r, record_l))
            # delete cache
            torch.cuda.empty_cache()
            gc.collect()

        return mask_adv
