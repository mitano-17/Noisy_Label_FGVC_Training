#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import moco.loader
import moco.builder
import moco.utils

from tqdm import tqdm


def train(train_loader, model, criterion, optimizer, epoch, args):
    
    # switch to train mode
    model.train()

    pbar = tqdm(train_loader, desc=f'Epoch {epoch} Training')

    # Initialize running averages
    running_loss = 0.0
    running_top1 = 0.0
    running_top5 = 0.0
    num_batches = 0

    #for i, (images, _) in enumerate(train_loader):
    for i, (images, _) in enumerate(pbar):

        if args.gpu is not None:
            images[0] = images[0].cuda(args.gpu, non_blocking=True)
            images[1] = images[1].cuda(args.gpu, non_blocking=True)

        # compute output
        output, target = model(im_q=images[0], im_k=images[1])
        loss = criterion(output, target)

        # acc1/acc5 are (K+1)-way contrast classifier accuracy
        # measure accuracy and record loss
        acc1, acc5 = moco.utils.accuracy(output, target, topk=(1, 5))
        
        batch_size = images[0].size(0)
        loss_value = loss.item()
        acc1_value = acc1[0].item()
        acc5_value = acc5[0].item()

        # Update running averages
        running_loss += loss_value
        running_top1 += acc1_value
        running_top5 += acc5_value
        num_batches += 1

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update progress bar
        pbar.set_postfix({
            'Loss': f'{running_loss/num_batches:.4f}',
            'Top1': f'{running_top1/num_batches:.2f}%',
            'Top5': f'{running_top5/num_batches:.2f}%',
            'LR': f'{optimizer.param_groups[0]["lr"]:.2e}'
        })

    current_lr = optimizer.param_groups[0]["lr"]
    avg_loss = running_loss / num_batches
    avg_top1 = running_top1 / num_batches
    avg_top5 = running_top5 / num_batches

    return avg_loss, avg_top1, avg_top5, current_lr