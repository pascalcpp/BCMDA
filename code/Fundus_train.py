import argparse
import logging
import os
import random
import shutil
import sys
from typing import Iterable

import numpy as np
import torch.backends.cudnn as cudnn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from medpy.metric import binary
from networks.unet_model_avg import UNet
from dataloaders.dataloader import FundusSegmentation
import dataloaders.custom_transforms as tr
from utils import losses, metrics, ramps, util
from torch.cuda.amp import autocast, GradScaler
import contextlib
import torch.nn.functional as F
from einops import rearrange
import torch


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='fundus', choices=['fundus', 'prostate'])
parser.add_argument("--save_name", type=str, default="BCMDA", help="experiment_name")
parser.add_argument("--overwrite", action='store_true')
parser.add_argument("--model", type=str, default="unet", help="model_name")
parser.add_argument("--max_iterations", type=int, default=30000, help="maximum epoch number to train")
parser.add_argument('--num_eval_iter', type=int, default=500)
parser.add_argument("--deterministic", type=int, default=1, help="whether use deterministic training")
parser.add_argument("--base_lr", type=float, default=0.03, help="segmentation network learning rate")
parser.add_argument("--seed", type=int, default=1337, help="random seed")
parser.add_argument("--gpu", type=str, default='0')
parser.add_argument("--threshold", type=float, default=0.95, help="confidence threshold for using pseudo-labels", )

parser.add_argument('--amp', type=int, default=1, help='use mixed precision training or not')

parser.add_argument("--label_bs", type=int, default=4, help="labeled_batch_size per gpu")
parser.add_argument("--unlabel_bs", type=int, default=4)
parser.add_argument("--test_bs", type=int, default=1)
parser.add_argument('--domain_num', type=int, default=4)
parser.add_argument('--lb_domain', type=int, default=1)
parser.add_argument('--lb_num', type=int, default=20)
# costs
parser.add_argument("--ema_decay", type=float, default=0.99, help="ema_decay")
parser.add_argument("--consistency", type=float, default=1.0, help="consistency")
parser.add_argument("--consistency_rampup", type=float, default=200.0, help="consistency_rampup")

parser.add_argument("--cutmix_prob", default=1.0, type=float)
parser.add_argument("--fix_r", default=0.75, type=float)
parser.add_argument("--beta_a", default=0.70, type=float)
parser.add_argument("--corr_resolution", default=64, type=int)
parser.add_argument("--temp", default=0.05, type=float)
parser.add_argument('--data_path', type=str, default='../data/Fundus')

args = parser.parse_args()



def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    # teacher network: ema_model
    # student network: model
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)

    return alpha


def cycle(iterable: Iterable):
    """Make an iterator returning elements from the iterable.

    .. note::
        **DO NOT** use `itertools.cycle` on `DataLoader(shuffle=True)`.\n
        Because `itertools.cycle` saves a copy of each element, batches are shuffled only at the first epoch. \n
        See https://docs.python.org/3/library/itertools.html#itertools.cycle for more details.
    """
    while True:
        for x in iterable:
            yield x


def generate_new_image_3c(last_fts1, last_fts2, image1, image2, w=64, size=256):
    last_fts1 = F.interpolate(last_fts1.detach(), (w, w), mode='bilinear', align_corners=True)
    last_fts2 = F.interpolate(last_fts2.detach(), (w, w), mode='bilinear', align_corners=True)
    image1 = F.interpolate(image1, (w, w), mode='bilinear', align_corners=True)
    image2 = F.interpolate(image2, (w, w), mode='bilinear', align_corners=True)
    image1 = rearrange(image1, 'n c h w -> n c (h w)')
    image2 = rearrange(image2, 'n c h w -> n c (h w)')

    f1 = rearrange(last_fts1, 'n c h w -> n c (h w)')
    f2 = rearrange(last_fts2, 'n c h w -> n c (h w)')
    corr_map_1_2 = torch.matmul(f1.transpose(1, 2), f2) / torch.sqrt(torch.tensor(f1.shape[1]).float())

    corr_map_2_1 = corr_map_1_2.transpose(1, 2).clone()

    corr_map_1_2 = F.softmax(corr_map_1_2, dim=-1)
    corr_map_2_1 = F.softmax(corr_map_2_1, dim=-1)

    new_image1 = rearrange(torch.matmul(image2, corr_map_2_1), 'n c (h w) -> n c h w', h=w, w=w)
    new_image2 = rearrange(torch.matmul(image1, corr_map_1_2), 'n c (h w) -> n c h w', h=w, w=w)

    new_image1 = torch.clip(new_image1, -1.0, 1.0)
    new_image2 = torch.clip(new_image2, -1.0, 1.0)

    new_image1 = F.interpolate(new_image1.detach(), (size, size), mode='bilinear', align_corners=True)
    new_image2 = F.interpolate(new_image2.detach(), (size, size), mode='bilinear', align_corners=True)

    new_image1 = torch.clip(new_image1, -1.0, 1.0)
    new_image2 = torch.clip(new_image2, -1.0, 1.0)

    return new_image1, new_image2
def obtain_cutmix_box(img_size, p=0.5, size_min=0.02, size_max=0.4, ratio_1=0.3, ratio_2=1 / 0.3):
    mask = torch.zeros(img_size, img_size).cuda()
    if random.random() > p:
        return mask

    size = np.random.uniform(size_min, size_max) * img_size * img_size
    while True:
        ratio = np.random.uniform(ratio_1, ratio_2)
        cutmix_w = int(np.sqrt(size / ratio))
        cutmix_h = int(np.sqrt(size * ratio))
        x = np.random.randint(0, img_size)
        y = np.random.randint(0, img_size)

        if x + cutmix_w <= img_size and y + cutmix_h <= img_size:
            break

    mask[y:y + cutmix_h, x:x + cutmix_w] = 1

    return mask


part = ['cup', 'disc']
dataset = FundusSegmentation
n_part = len(part)
dice_calcu = {'fundus': metrics.dice_coeff_2label, 'prostate': metrics.dice_coeff, 'MNMS': metrics.dice_coeff_3label}

@torch.no_grad()
def test_all(args, model, test_dataloader, epoch):
    model.eval()
    val_dice = [0.0] * n_part
    val_dc, val_jc, val_hd, val_asd = [0.0] * n_part, [0.0] * n_part, [0.0] * n_part, [0.0] * n_part
    domain_num = len(test_dataloader)
    num = 0
    for i in range(domain_num):
        cur_dataloader = test_dataloader[i]
        domain_val_dice = [0.0] * n_part
        domain_val_dc, domain_val_jc, domain_val_hd, domain_val_asd = [0.0] * n_part, [0.0] * n_part, [0.0] * n_part, [
            0.0] * n_part
        domain_code = i + 1
        for batch_num, sample in enumerate(cur_dataloader):
            data = sample['image'].cuda()

            mask = sample['label'].cuda()
            cup_mask = mask.eq(0).float()
            disc_mask = mask.le(128).float()
            mask = torch.cat((cup_mask.unsqueeze(1), disc_mask.unsqueeze(1)), dim=1)

            res_test = model(data)

            output_linear = model.classify_linear(res_test['last_fts'])

            output = output_linear
            p_linear = torch.sigmoid(output_linear)
            pred_prob = p_linear

            pred_prob = pred_prob.cpu()
            mask = mask.cpu()
            output = output.cpu()
            pred_label = pred_prob.ge(0.5)
            pred_onehot = pred_label.clone()
            mask_onehot = mask.clone()

            dice = dice_calcu[args.dataset](np.asarray(pred_label), mask)


            dc, jc, hd, asd = [0.0] * n_part, [0.0] * n_part, [0.0] * n_part, [0.0] * n_part
            for j in range(len(data)):
                for i, p in enumerate(part):
                    dc[i] += binary.dc(np.asarray(pred_onehot[j, i], dtype=bool),
                                       np.asarray(mask_onehot[j, i], dtype=bool))
                    jc[i] += binary.jc(np.asarray(pred_onehot[j, i], dtype=bool),
                                       np.asarray(mask_onehot[j, i], dtype=bool))
                    if pred_onehot[j, i].float().sum() < 1e-4:
                        hd[i] += 100
                        asd[i] += 100
                    else:
                        hd[i] += binary.hd95(np.asarray(pred_onehot[j, i], dtype=bool),
                                             np.asarray(mask_onehot[j, i], dtype=bool))
                        asd[i] += binary.asd(np.asarray(pred_onehot[j, i], dtype=bool),
                                             np.asarray(mask_onehot[j, i], dtype=bool))
            for i, p in enumerate(part):
                dc[i] /= len(data)
                jc[i] /= len(data)
                hd[i] /= len(data)
                asd[i] /= len(data)
            for i in range(len(domain_val_dice)):
                domain_val_dice[i] += dice[i]
                domain_val_dc[i] += dc[i]
                domain_val_jc[i] += jc[i]
                domain_val_hd[i] += hd[i]
                domain_val_asd[i] += asd[i]

        for i in range(len(domain_val_dice)):
            domain_val_dice[i] /= len(cur_dataloader)
            val_dice[i] += domain_val_dice[i]
            domain_val_dc[i] /= len(cur_dataloader)
            val_dc[i] += domain_val_dc[i]
            domain_val_jc[i] /= len(cur_dataloader)
            val_jc[i] += domain_val_jc[i]
            domain_val_hd[i] /= len(cur_dataloader)
            val_hd[i] += domain_val_hd[i]
            domain_val_asd[i] /= len(cur_dataloader)
            val_asd[i] += domain_val_asd[i]
        text = 'domain%d epoch %d :' % (domain_code, epoch)
        text += '\n\t'
        for n, p in enumerate(part):
            text += 'val_%s_dice: %f, ' % (p, domain_val_dice[n])
        text += '\n\t'
        for n, p in enumerate(part):
            text += 'val_%s_dc: %f, ' % (p, domain_val_dc[n])

        text += '\t'
        for n, p in enumerate(part):
            text += 'val_%s_jc: %f, ' % (p, domain_val_jc[n])
        text += '\n\t'
        for n, p in enumerate(part):
            text += 'val_%s_hd: %f, ' % (p, domain_val_hd[n])
        text += '\t'
        for n, p in enumerate(part):
            text += 'val_%s_asd: %f, ' % (p, domain_val_asd[n])
        logging.info(text)

    model.train()
    for i in range(len(val_dice)):
        val_dice[i] /= domain_num
        val_dc[i] /= domain_num
        val_jc[i] /= domain_num
        val_hd[i] /= domain_num
        val_asd[i] /= domain_num
    text = 'epoch %d :' % (epoch)
    text += '\n\t'
    avg = 0.0
    for n, p in enumerate(part):
        text += 'val_%s_dice: %f, ' % (p, val_dice[n])
        avg += val_dice[n]

    avg = avg / len(val_dice)
    text += 'val_avg_dice: %f, ' % (avg)

    text += '\n\t'
    avg = 0.0
    for n, p in enumerate(part):
        text += 'val_%s_dc: %f, ' % (p, val_dc[n])
        avg += val_dc[n]
    avg = avg / len(val_dc)
    text += 'val_avg_dc: %f, ' % (avg)
    text += '\t'

    avg = 0.0
    for n, p in enumerate(part):
        text += 'val_%s_jc: %f, ' % (p, val_jc[n])
        avg += val_jc[n]
    avg = avg / len(val_jc)
    text += 'val_avg_jc: %f, ' % (avg)
    text += '\n\t'

    avg = 0.0
    for n, p in enumerate(part):
        text += 'val_%s_hd: %f, ' % (p, val_hd[n])
        avg += val_hd[n]
    avg = avg / len(val_hd)
    text += 'val_avg_hd: %f, ' % (avg)
    text += '\t'

    avg = 0.0
    for n, p in enumerate(part):
        text += 'val_%s_asd: %f, ' % (p, val_asd[n])
        avg += val_asd[n]
    avg = avg / len(val_asd)
    text += 'val_avg_asd: %f, ' % (avg)
    logging.info(text)
    return val_dc




def train(args, snapshot_path):
    writer = SummaryWriter(snapshot_path + '/log')

    base_lr = args.base_lr
    num_channels = 3
    patch_size = 256
    num_classes = 2
    min_v, max_v = 0.5, 1.5
    fillcolor = 255
    max_iterations = args.max_iterations

    weak = transforms.Compose([tr.RandomScaleCrop(patch_size),
                               tr.RandomScaleRotate(fillcolor=fillcolor),
                               tr.RandomHorizontalFlip(),
                               tr.elastic_transform(),
                               ])

    strong = transforms.Compose([
        tr.Brightness(min_v, max_v),
        tr.Contrast(min_v, max_v),
        tr.GaussianBlur(kernel_size=int(0.1 * patch_size), num_channels=num_channels),
    ])

    normal_toTensor = transforms.Compose([
        tr.Normalize_tf(),
        tr.ToTensor()
    ])

    domain_num = args.domain_num
    domain = list(range(1, domain_num + 1))
    domain_len = [50, 99, 320, 320]
    lb_domain = args.lb_domain
    data_num = domain_len[lb_domain - 1]
    lb_num = args.lb_num
    lb_idxs = list(range(lb_num))
    unlabeled_idxs = list(range(lb_num, data_num))
    test_dataset = []
    test_dataloader = []
    lb_dataset = dataset(base_dir=train_data_path, phase='train', splitid=lb_domain, domain=[lb_domain],
                         selected_idxs=lb_idxs, weak_transform=weak, normal_toTensor=normal_toTensor)
    ulb_dataset = dataset(base_dir=train_data_path, phase='train', splitid=lb_domain, domain=domain,
                          selected_idxs=unlabeled_idxs, weak_transform=weak, strong_tranform=strong,
                          normal_toTensor=normal_toTensor)
    for i in range(1, domain_num + 1):
        cur_dataset = dataset(base_dir=train_data_path, phase='test', splitid=-1, domain=[i],
                              normal_toTensor=normal_toTensor)
        test_dataset.append(cur_dataset)

    nws = 2
    lb_dataloader = cycle(
        DataLoader(lb_dataset, batch_size=args.label_bs, shuffle=True, num_workers=nws, pin_memory=True,
                   drop_last=True))

    ulb_dataloader = cycle(
        DataLoader(ulb_dataset, batch_size=args.unlabel_bs, shuffle=True, num_workers=nws, pin_memory=True,
                   drop_last=True))
    for i in range(0, domain_num):
        cur_dataloader = DataLoader(test_dataset[i], batch_size=args.test_bs, shuffle=False, num_workers=0,
                                    pin_memory=True)
        test_dataloader.append(cur_dataloader)

    def create_model(ema=False):
        # Network definition
        if args.model == 'unet':
            model = UNet(n_channels=num_channels, n_classes=num_classes, temp=args.temp)
        if ema:
            for param in model.parameters():
                param.detach_()
        return model.cuda()

    model = create_model()
    ema_model = create_model(ema=True)

    iter_num = 0
    start_epoch = 0

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    # set to train
    ce_loss = torch.nn.BCEWithLogitsLoss(reduction='none')
    softmax, sigmoid, multi = False, True, True
    dice_loss = losses.DiceLossWithMask(num_classes)


    max_epoch = max_iterations // args.num_eval_iter
    stu_best_dice = [0.0] * n_part
    stu_best_dice_iter = [-1] * n_part
    stu_best_avg_dice = 0.0
    stu_best_avg_dice_iter = -1
    stu_dice_of_best_avg = [0.0] * n_part

    logging.info("{} iterations per epoch".format(args.num_eval_iter))
    logging.info("total {} epoch".format(max_epoch))

    iter_num = int(iter_num)
    threshold = args.threshold

    scaler = GradScaler()
    amp_cm = autocast if args.amp else contextlib.nullcontext


    for epoch_num in range(start_epoch, max_epoch):
        loss_ul1_avg = util.AverageMeter()
        loss_lu1_avg = util.AverageMeter()
        loss_ul2_avg = util.AverageMeter()
        loss_lu2_avg = util.AverageMeter()
        loss_avg = util.AverageMeter()
        mask_avg = util.AverageMeter()

        model.train()
        ema_model.train()
        p_bar = tqdm(range(args.num_eval_iter))
        p_bar.set_description(f'No. {epoch_num + 1}')
        for i_batch in range(1, args.num_eval_iter + 1):
            lb_sample = next(lb_dataloader)
            ulb_sample = next(ulb_dataloader)
            lb_x_w, lb_y = lb_sample['image'], lb_sample['label']
            ulb_x_w, ulb_y = ulb_sample['image'], ulb_sample['label']
            lb_dc, ulb_dc = lb_sample['dc'].cuda(), ulb_sample['dc'].cuda()
            ulb_x_s = ulb_sample['strong_aug'].cuda()
            lb_x_w, lb_y, ulb_x_w, ulb_y = lb_x_w.cuda(), lb_y.cuda(), ulb_x_w.cuda(), ulb_y.cuda()

            lb_cup_label = lb_y.eq(0).float()  # == 0
            lb_disc_label = lb_y.le(128).float()  # <= 128
            lb_mask = torch.cat((lb_cup_label.unsqueeze(1), lb_disc_label.unsqueeze(1)), dim=1)

            with amp_cm():
                with torch.no_grad():
                    label_box1 = torch.stack(
                        [obtain_cutmix_box(img_size=patch_size, p=args.cutmix_prob) for i in range(len(ulb_x_w))],
                        dim=0)
                    img_box1 = label_box1.unsqueeze(1)
                    label_box1 = label_box1.unsqueeze(1)

                    label_box2 = torch.stack(
                        [obtain_cutmix_box(img_size=patch_size, p=args.cutmix_prob) for i in range(len(ulb_x_w))],
                        dim=0)
                    img_box2 = label_box2.unsqueeze(1)
                    label_box2 = label_box2.unsqueeze(1)

                    consistency_weight = get_current_consistency_weight(
                        iter_num // (args.max_iterations / args.consistency_rampup))

                    res_ulb_x_w = ema_model(ulb_x_w)
                    last_fts_ulb_x_w_ema = res_ulb_x_w['last_fts']
                    logits_ulb_x_w_sim = ema_model.classify_sim_avg(last_fts_ulb_x_w_ema)
                    logits_ulb_x_w_linear = ema_model.classify_linear(last_fts_ulb_x_w_ema)

                    prob_ulb_sim = logits_ulb_x_w_sim.sigmoid()
                    prob_ulb_linear = logits_ulb_x_w_linear.sigmoid()
                    better_mask_ulb = (prob_ulb_sim > threshold).float()
                    prob_ = prob_ulb_sim * better_mask_ulb + prob_ulb_linear * (1 - better_mask_ulb)

                    res_lb_x_w = ema_model(lb_x_w)
                    last_fts_lb_x_w_ema = res_lb_x_w['last_fts']
                    same_domain = (lb_dc == ulb_dc).bool()

                with torch.no_grad():

                    new_lb_x_w, new_ulb_x_w = generate_new_image_3c(last_fts_lb_x_w_ema, last_fts_ulb_x_w_ema,
                                                                               lb_x_w.clone(), ulb_x_w.clone(),
                                                                               args.corr_resolution, patch_size)

                    new_lb_x_w, new_ulb_x_w = new_lb_x_w.float(), new_ulb_x_w.float()

                    fix_r = args.fix_r
                    beta_a = args.beta_a
                    upper = min(fix_r, iter_num / max_iterations)
                    mix_r = np.random.beta(beta_a, beta_a) * upper
                    lb_zero_mask = (lb_x_w == -1.0).float()
                    ulb_zero_mask = (ulb_x_w == -1.0).float()

                    mix_lb_fix_w = (1 - fix_r) * lb_x_w + fix_r * new_lb_x_w
                    mix_lb_fix_w = (1 - lb_zero_mask) * mix_lb_fix_w + lb_zero_mask * lb_x_w

                    mix_ulb_fix_w = (1 - fix_r) * ulb_x_w + fix_r * new_ulb_x_w
                    mix_ulb_fix_w = (1 - ulb_zero_mask) * mix_ulb_fix_w + ulb_zero_mask * ulb_x_w

                    mix_lb_w = (1 - mix_r) * lb_x_w + mix_r * new_lb_x_w
                    mix_lb_w = (1 - lb_zero_mask) * mix_lb_w + lb_zero_mask * lb_x_w

                    res_mix_ulb_fix_w = ema_model(mix_ulb_fix_w)
                    logits_mix_ulb_fix_w_sim = ema_model.classify_sim_avg(res_mix_ulb_fix_w['last_fts'])
                    logits_mix_ulb_fix_w_linear = ema_model.classify_linear(res_mix_ulb_fix_w['last_fts'])

                    prob_mix_ulb_fix_w_sim = logits_mix_ulb_fix_w_sim.sigmoid()
                    prob_mix_ulb_fix_w_linear = logits_mix_ulb_fix_w_linear.sigmoid()
                    better_mask_mix_ulb = (prob_mix_ulb_fix_w_sim > threshold).float()
                    prob_mix_ulb_fix_w = prob_mix_ulb_fix_w_sim * better_mask_mix_ulb + prob_mix_ulb_fix_w_linear * (
                            1 - better_mask_mix_ulb)

                    stable_prob = (prob_mix_ulb_fix_w + prob_) / 2.0
                    stable_prob[same_domain] = prob_[same_domain]
                    stable_label = stable_prob.ge(0.5).float()
                    stable_mask = stable_prob.ge(threshold).float() + stable_prob.le(1 - threshold).float()

                    mask_ul1, mask_lu1 = stable_mask.clone(), stable_mask.clone()
                    pseudo_label_ul1 = (stable_label * (1 - label_box1) + lb_mask * label_box1).long()
                    mask_ul1[img_box1.expand(mask_ul1.shape) == 1] = 1
                    pseudo_label_lu1 = (lb_mask * (1 - label_box1) + stable_label * label_box1).long()
                    mask_lu1[img_box1.expand(mask_lu1.shape) == 0] = 1

                    pseudo_label_ul1 = pseudo_label_ul1.float()
                    pseudo_label_lu1 = pseudo_label_lu1.float()

                    mask_ul2, mask_lu2 = stable_mask.clone(), stable_mask.clone()
                    pseudo_label_ul2 = (stable_label * (1 - label_box2) + lb_mask * label_box2).long()
                    mask_ul2[img_box2.expand(mask_ul2.shape) == 1] = 1
                    pseudo_label_lu2 = (lb_mask * (1 - label_box2) + stable_label * label_box2).long()
                    mask_lu2[img_box2.expand(mask_lu2.shape) == 0] = 1

                    pseudo_label_ul2 = pseudo_label_ul2.float()
                    pseudo_label_lu2 = pseudo_label_lu2.float()

                    mix_lb_w[same_domain] = lb_x_w[same_domain]
                    mix_lb_fix_w[same_domain] = lb_x_w[same_domain]
                    mix_ulb_fix_w[same_domain] = ulb_x_w[same_domain]

                x_ul_1 = mix_ulb_fix_w * (1 - img_box1) + mix_lb_fix_w * img_box1
                x_lu_1 = mix_lb_fix_w * (1 - img_box1) + mix_ulb_fix_w * img_box1

                x_ul_2 = ulb_x_s * (1 - img_box2) + mix_lb_w * img_box2
                x_lu_2 = mix_lb_w * (1 - img_box2) + ulb_x_s * img_box2

                res_x_ul_1 = model(x_ul_1)
                res_x_lu_1 = model(x_lu_1)

                res_x_ul_2 = model(x_ul_2)
                res_x_lu_2 = model(x_lu_2)

                last_fts_x_ul_1, last_fts_x_ul_2 = res_x_ul_1['last_fts'], res_x_ul_2['last_fts']
                last_fts_x_lu_1, last_fts_x_lu_2 = res_x_lu_1['last_fts'], res_x_lu_2['last_fts']

                logits_x_ul_1_sim = model.classify_sim1(last_fts_x_ul_1, consistency_weight)
                logits_x_ul_1_linear = model.classify_linear(last_fts_x_ul_1)

                logits_x_lu_1_sim = model.classify_sim1(last_fts_x_lu_1, consistency_weight)
                logits_x_lu_1_linear = model.classify_linear(last_fts_x_lu_1)

                logits_x_ul_2_sim = model.classify_sim2(last_fts_x_ul_2, consistency_weight)
                logits_x_ul_2_linear = model.classify_linear(last_fts_x_ul_2)

                logits_x_lu_2_sim = model.classify_sim2(last_fts_x_lu_2, consistency_weight)
                logits_x_lu_2_linear = model.classify_linear(last_fts_x_lu_2)

                loss_ul_1_sim = (ce_loss(logits_x_ul_1_sim, pseudo_label_ul1) * mask_ul1.squeeze(1)).mean() + \
                                dice_loss(logits_x_ul_1_sim, pseudo_label_ul1.unsqueeze(1), mask=mask_ul1,
                                          softmax=softmax,
                                          sigmoid=sigmoid, multi=multi)
                loss_ul_1_linear = (ce_loss(logits_x_ul_1_linear, pseudo_label_ul1) * mask_ul1.squeeze(1)).mean() + \
                                   dice_loss(logits_x_ul_1_linear, pseudo_label_ul1.unsqueeze(1), mask=mask_ul1,
                                             softmax=softmax,
                                             sigmoid=sigmoid, multi=multi)
                loss_ul_1 = (loss_ul_1_sim + loss_ul_1_linear) / 2.0

                loss_lu_1_sim = (ce_loss(logits_x_lu_1_sim, pseudo_label_lu1) * mask_lu1.squeeze(1)).mean() + \
                                dice_loss(logits_x_lu_1_sim, pseudo_label_lu1.unsqueeze(1), mask=mask_lu1,
                                          softmax=softmax,
                                          sigmoid=sigmoid, multi=multi)
                loss_lu_1_linear = (ce_loss(logits_x_lu_1_linear, pseudo_label_lu1) * mask_lu1.squeeze(1)).mean() + \
                                   dice_loss(logits_x_lu_1_linear, pseudo_label_lu1.unsqueeze(1), mask=mask_lu1,
                                             softmax=softmax,
                                             sigmoid=sigmoid, multi=multi)
                loss_lu_1 = (loss_lu_1_sim + loss_lu_1_linear) / 2.0

                loss_ul_2_sim = (ce_loss(logits_x_ul_2_sim, pseudo_label_ul2) * mask_ul2.squeeze(1)).mean() + \
                                dice_loss(logits_x_ul_2_sim, pseudo_label_ul2.unsqueeze(1), mask=mask_ul2,
                                          softmax=softmax,
                                          sigmoid=sigmoid, multi=multi)
                loss_ul_2_linear = (ce_loss(logits_x_ul_2_linear, pseudo_label_ul2) * mask_ul2.squeeze(1)).mean() + \
                                   dice_loss(logits_x_ul_2_linear, pseudo_label_ul2.unsqueeze(1), mask=mask_ul2,
                                             softmax=softmax,
                                             sigmoid=sigmoid, multi=multi)

                loss_ul_2 = (loss_ul_2_sim + loss_ul_2_linear) / 2.0

                loss_lu_2_sim = (ce_loss(logits_x_lu_2_sim, pseudo_label_lu2) * mask_lu2.squeeze(1)).mean() + \
                                dice_loss(logits_x_lu_2_sim, pseudo_label_lu2.unsqueeze(1), mask=mask_lu2,
                                          softmax=softmax,
                                          sigmoid=sigmoid, multi=multi)
                loss_lu_2_linear = (ce_loss(logits_x_lu_2_linear, pseudo_label_lu2) * mask_lu2.squeeze(1)).mean() + \
                                   dice_loss(logits_x_lu_2_linear, pseudo_label_lu2.unsqueeze(1), mask=mask_lu2,
                                             softmax=softmax,
                                             sigmoid=sigmoid, multi=multi)
                loss_lu_2 = (loss_lu_2_sim + loss_lu_2_linear) / 2.0

                loss_1 = (loss_ul_1 + loss_lu_1) / 2.0
                loss_2 = (loss_ul_2 + loss_lu_2) / 2.0
                loss = loss_1 + loss_2

            optimizer.zero_grad()

            if args.amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            # update ema model
            update_ema_variables(model, ema_model, args.ema_decay, iter_num)

            loss_avg.update(loss.item())
            loss_ul1_avg.update(loss_ul_1.item())
            loss_lu1_avg.update(loss_lu_1.item())
            loss_ul2_avg.update(loss_ul_2.item())
            loss_lu2_avg.update(loss_lu_2.item())
            mask_avg.update(stable_mask.mean())

            # update learning rate
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1

            if p_bar is not None:
                p_bar.update()

            p_bar.set_description(
                'iteration %d: loss:%.4f, loss_ul_1:%.4f, loss_lu_1:%.4f, loss_ul_2:%.4f, loss_lu_2:%.4f, cons_w:%.4f, lr:%.4f, mask_ratio:%.4f'
                % (iter_num, loss_avg.avg, loss_ul1_avg.avg, loss_lu1_avg.avg, loss_ul2_avg.avg, loss_lu2_avg.avg,
                   consistency_weight, lr_,
                   mask_avg.avg
                   ))

        if p_bar is not None:
            p_bar.close()
        logging.info(
                'iteration %d: loss:%.4f, loss_ul_1:%.4f, loss_lu_1:%.4f, loss_ul_2:%.4f, loss_lu_2:%.4f, cons_w:%.4f, lr:%.4f, mask_ratio:%.4f'
                % (iter_num, loss_avg.avg, loss_ul1_avg.avg, loss_lu1_avg.avg, loss_ul2_avg.avg, loss_lu2_avg.avg,
                   consistency_weight, lr_,
                   mask_avg.avg
                   ))

        logging.info('test stu model')
        stu_val_dice = test_all(args, model, test_dataloader, epoch_num + 1)
        text = ''
        for n, p in enumerate(part):
            if stu_val_dice[n] > stu_best_dice[n]:
                stu_best_dice[n] = stu_val_dice[n]
                stu_best_dice_iter[n] = iter_num
            text += 'stu_val_%s_best_dice: %f at %d iter' % (p, stu_best_dice[n], stu_best_dice_iter[n])
            text += ', '
        if sum(stu_val_dice) / len(stu_val_dice) > stu_best_avg_dice:
            stu_best_avg_dice = sum(stu_val_dice) / len(stu_val_dice)
            stu_best_avg_dice_iter = iter_num
            for n, p in enumerate(part):
                stu_dice_of_best_avg[n] = stu_val_dice[n]
            save_text = "{}_avg_dice_best_model.pth".format(args.model)
            save_best = os.path.join(snapshot_path, save_text)
            logging.info('save cur best avg model to {}'.format(save_best))
            torch.save(model.state_dict(), save_best)
        text += 'val_best_avg_dice: %f at %d iter' % (stu_best_avg_dice, stu_best_avg_dice_iter)
        if n_part > 1:
            for n, p in enumerate(part):
                text += ', %s_dice: %f' % (p, stu_dice_of_best_avg[n])
        logging.info(text)

    writer.close()


if __name__ == "__main__":
    snapshot_path = "../model/" + args.dataset + "/" + args.save_name + '_' + str(args.lb_domain) + '_' + str(
        args.fix_r) + '_' + str(args.beta_a) + '_' + str(args.seed) + '_' + str(args.lb_num) + "/"


    train_data_path = args.data_path
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    elif not args.overwrite:
        raise Exception('file {} is exist!'.format(snapshot_path))
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code', shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    cmd = " ".join(["python"] + sys.argv)
    logging.info(cmd)
    logging.info(str(args))

    train(args, snapshot_path)
