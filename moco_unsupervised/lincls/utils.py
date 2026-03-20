#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
import shutil
import torch
import torch.nn as nn
import torchvision.models as models
import pandas as pd


def create_model(args):

    print("=> creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch]()

    return model


def freeze(model):

	# freeze all layers but the last fc
    for name, param in model.named_parameters():
        if name not in ['fc.weight', 'fc.bias']:
            param.requires_grad = False

    # init the fc layer
    model.fc.weight.data.normal_(mean=0.0, std=0.01)
    model.fc.bias.data.zero_()

    return model


def load_pretrained(args, model):
    if os.path.isfile(args.pretrained):
        print("=> loading checkpoint '{}'".format(args.pretrained))
        checkpoint = torch.load(args.pretrained, weights_only=False, map_location="cpu")

        # rename moco pre-trained keys
        state_dict = checkpoint['state_dict']

        for k in list(state_dict.keys()):
            # retain only encoder_q up to before the embedding layer
            if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                # remove prefix
                state_dict[k[len("module.encoder_q."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]

        args.start_epoch = 0
        msg = model.load_state_dict(state_dict, strict=False)
        assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}

        print("=> loaded pre-trained model '{}'".format(args.pretrained))
    else:
        print("=> no checkpoint found at '{}'".format(args.pretrained))


def moco_func(args, model):

    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    # optimize only the linear classifier
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))

    assert len(parameters) == 2  # fc.weight, fc.bias

    optimizer = torch.optim.SGD(parameters, args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    return criterion, optimizer


def resume_checkpoint(args, model, optimizer):

    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))

        if args.gpu is None:
            checkpoint = torch.load(args.resume, weights_only=False)

        else:
            # Map model to be loaded to specified single gpu.
            loc = 'cuda:{}'.format(args.gpu)
            checkpoint = torch.load(args.resume, weights_only=False, map_location=loc)

        args.start_epoch = checkpoint['epoch']
        best_acc1 = checkpoint['best_acc1']

        if args.gpu is not None:
            # best_acc1 may be from a checkpoint from a different GPU
            best_acc1 = best_acc1.to(args.gpu)

        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.resume, checkpoint['epoch']))

    else:
        print("=> no checkpoint found at '{}'".format(args.resume))



def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', directory='./moco_results'):

    checkpoint_path = os.path.join(directory, filename)
    torch.save(state, checkpoint_path)

    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def sanity_check(state_dict, pretrained_weights):
    """
    Linear classifier should not change any weights other than the linear layer.
    This sanity check asserts nothing wrong happens (e.g., BN stats updated).
    """
    print("=> loading '{}' for sanity check".format(pretrained_weights))
    checkpoint = torch.load(pretrained_weights, map_location="cpu")
    state_dict_pre = checkpoint['state_dict']

    for k in list(state_dict.keys()):
        # only ignore fc layer
        if 'fc.weight' in k or 'fc.bias' in k:
            continue

        # name in pretrained model
        k_pre = 'module.encoder_q.' + k[len('module.'):] \
            if k.startswith('module.') else 'module.encoder_q.' + k

        assert ((state_dict[k].cpu() == state_dict_pre[k_pre]).all()), \
            '{} is changed in linear classifier training.'.format(k)

    print("=> sanity check passed.")


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    for milestone in args.schedule:
        lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def init_csv_log(csv_log_path):
    """Initialize CSV file with SimCLR-specific headers"""
    headers = [
        'epoch',  
        'train loss', 
        'val loss', 
        'train top1',  
        'train top5',
        'val top1',  
        'val top5',  
        'learning_rate'
    ]
    
    # Create empty DataFrame with headers
    df = pd.DataFrame(columns=headers)
    df.to_csv(csv_log_path, index=False)

    print(f"CSV logging initialized at {csv_log_path}")

def log_metrics(log_data, directory=''):
    """Log metrics to CSV file"""
    
    # Append to CSV
    df = pd.DataFrame([log_data])

    df.to_csv(directory, mode='a', header=False, index=False)