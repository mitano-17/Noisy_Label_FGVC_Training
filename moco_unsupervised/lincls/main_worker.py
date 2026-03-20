#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import builtins
import os
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

from lincls.train import train
from lincls.validate import validate
from lincls.load_dataset import load_dataset

import lincls.utils


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
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
    model = lincls.utils.create_model(args)
    # freeze model layers
    model = lincls.utils.freeze(model)

    # load from pre-trained, before DistributedDataParallel constructor
    if args.pretrained:
        lincls.utils.load_pretrained(args, model)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
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
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    csv_log_path = os.path.join(args.save_dir, 'linear_classification_metrics.csv')
    # initialize metrics file
    lincls.utils.init_csv_log(csv_log_path)
    # define loss function (criterion) and optimizer
    criterion, optimizer = lincls.utils.moco_func(args, model)

    # optionally resume from a checkpoint
    if args.resume:
        lincls.utils.resume_checkpoint(args, model, optimizer)

    cudnn.benchmark = True

    # Data loading code
    train_loader, train_sampler, val_loader = load_dataset(args)

    if args.evaluate:
        val_loss, val_top1, val_top5, val_current_lr = validate(val_loader, model, criterion, args)

        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        lincls.utils.adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train_loss, train_top1, train_top5, train_current_lr = train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        val_loss, val_top1, val_top5, val_current_lr = validate(val_loader, model, criterion, args)

        # remember best acc@1 and save checkpoint
        is_best = val_top1 > best_acc1
        best_acc1 = max(val_top1, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):

            print(f'Epoch {epoch} Summary - Loss: {val_loss:.4f}, Top1: {val_top1:.2f}%, Top5: {val_top5:.2f}%, LR: {val_current_lr:.2e}')

        	# log to csv
            lincls.utils.log_metrics({
                    'epoch': epoch,
                    'train loss': train_loss,
                    'val loss': val_loss,
                    'train top1': train_top1,
                    'train top5': train_top5,
                    'val top1': val_top1,
                    'val top5': val_top5,
                    'learning_rate': val_current_lr
                }, csv_log_path)

            if (epoch + 1) % args.save_freq == 0:
	            lincls.utils.save_checkpoint({
    	                'epoch': epoch + 1,
    	                'arch': args.arch,
    	                'state_dict': model.state_dict(),
    	                'best_acc1': best_acc1,
    	                'optimizer' : optimizer.state_dict()
                    }, is_best, filename='checkpoint_{:04d}.pth.tar'.format(epoch), directory=args.save_dir)

            if epoch == args.start_epoch:
                lincls.utils.sanity_check(model.state_dict(), args.pretrained)

    # prevent zombie process
    if args.distributed:
        dist.destroy_process_group()