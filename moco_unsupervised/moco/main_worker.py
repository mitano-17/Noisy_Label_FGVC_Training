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

from moco.load_dataset import load_dataset 

from moco.train_moco import train


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    model = moco.utils.create_model(args)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)

    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")


    csv_log_path = os.path.join(args.save_dir, 'training_metrics.csv')
    # initialize metrics file
    moco.utils.init_csv_log(csv_log_path)
    # define loss function (criterion) and optimizer
    criterion, optimizer = moco.utils.moco_func(args, model)

    # optionally resume from a checkpoint
    if args.resume:
        moco.utils.resume_checkpoint(args, model, optimizer)

    cudnn.benchmark = True

    # Data loading code ====================================================
    train_loader, train_sampler = load_dataset(args)    

    for epoch in range(args.start_epoch, args.epochs):

        if args.distributed:
            train_sampler.set_epoch(epoch)

        moco.utils.adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        avg_loss, avg_top1, avg_top5, current_lr = train(train_loader, model, criterion, optimizer, epoch, args)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):

            print(f'Epoch {epoch} Summary - Loss: {avg_loss:.4f}, Top1: {avg_top1:.2f}%, Top5: {avg_top5:.2f}%, LR: {current_lr:.2e}')

            # log to csv
            moco.utils.log_metrics({
                'epoch': epoch,  
                'loss': avg_loss,  
                'top1': avg_top1,  
                'top5': avg_top5,  
                'learning_rate': current_lr
            }, csv_log_path)

            if (epoch + 1) % args.save_freq == 0:
                moco.utils.save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                }, is_best=False, filename='checkpoint_{:04d}.pth.tar'.format(epoch), directory=args.save_dir)

    # prevent zombie process
    if args.distributed:
        dist.destroy_process_group()