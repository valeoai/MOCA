"""
Adapted from the iBOT[*] code: https://github.com/bytedance/ibot
[*] iBOT: Image BERT Pre-Training with Online Tokenizer, ICLR'22
"""

import os
import argparse
import json
import copy
import itertools
import torch
import torch.backends.cudnn as cudnn
import numpy as np

import util.misc as misc
from util.data_utils import prepare_imagenet_datasets
from util.model_utils import load_pretrained_encoder

from timm.utils import accuracy
from torch import nn
import models_vit
from pathlib import Path


def restart_from_checkpoint(ckp_path, models, optimizers, schedulers, run_variables=None):
    """
    Re-start from checkpoint
    """
    if not os.path.isfile(ckp_path):
        return
    print("Found checkpoint at {}".format(ckp_path))

    # open checkpoint file
    checkpoint = torch.load(ckp_path, map_location="cpu")
    msg = models.load_state_dict(checkpoint["state_dict"], strict=True)
    for idx, optim in enumerate(optimizers):
        msg = optim.load_state_dict(checkpoint["optimizers"][idx])
    for idx, sceduler in enumerate(schedulers):
        msg = sceduler.load_state_dict(checkpoint["schedulers"][idx])

    # re load variable important for the run
    if run_variables is not None:
        for var_name in run_variables:
            if var_name in checkpoint:
                run_variables[var_name] = checkpoint[var_name]


def eval_linear(args):
    misc.init_distributed_mode(args)
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))
    device = torch.device(args.device)
    cudnn.benchmark = True

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # ============ preparing data ... ============
    dataset_train, train_loader, dataset_val, val_loader = prepare_imagenet_datasets(
        args.data_path, misc.get_rank(), misc.get_world_size(), input_size=args.input_size, subset=args.subset,
        batch_size=args.batch_size, num_workers=args.num_workers, pin_mem=args.pin_mem, dist_eval=args.dist_eval)
    print(f"Data loaded for training: there are {len(dataset_train)} images.")
    print(f"Data loaded for validation: there are {len(dataset_val)} images.")


    # ============ building network ... ============
    args.avg_pooling = args.avgpool_patchtokens == 1
    model = models_vit.__dict__[args.arch](
        num_classes=0,
        drop_path_rate=0,
        global_pool=args.avg_pooling,
        fc_norm=args.use_fc_norm)
    embed_dim = model.embed_dim
    print(f"Model {args.arch} built.")

    # load weights to evaluate
    if args.pretrained_weights:
        load_pretrained_encoder(model, args.pretrained_weights, 
            use_teacher=args.eval_teacher, avg_pooling=args.avg_pooling,
            use_fc_norm=args.use_fc_norm)

    model.to(args.gpu)
    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    args.lrs = [base*n for base in [10**k for k in range(-4, 1)] for n in range(1, 10)]

    if not args.sweep_lr_only:
        args.wds = [0, 1e-6, 1e-5, 1e-4]
        args.optims = ['sgd']
    else:
        args.wds = [0]
        args.optims = ['sgd']
    args.permutes = list(itertools.product(args.lrs, args.wds, args.optims))
    feat_dim = embed_dim * (args.n_last_blocks * int(args.avgpool_patchtokens != 1) + int(args.avgpool_patchtokens > 0))

    linear_classifiers = nn.ModuleList()
    optimizers = []
    schedulers = []
    for pm in args.permutes:
        lr, wd, optim = pm
        linear_classifier = LinearClassifier(feat_dim, num_labels=args.num_labels)
        linear_classifier = linear_classifier.cuda()
        linear_classifier = nn.parallel.DistributedDataParallel(linear_classifier, device_ids=[args.gpu])
        linear_classifiers.append(linear_classifier)

        # set optimizer
        parameters = linear_classifier.parameters()
        assert optim == 'sgd'
        optimizer = torch.optim.SGD(
            parameters,
            lr * (args.batch_size * misc.get_world_size()) / 256., # linear scaling rule
            momentum=0.9,
            weight_decay=wd,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0)

        optimizers.append(optimizer)
        schedulers.append(scheduler)

    # Optionally resume from a checkpoint
    to_restore = {"epoch": 0, "best_acc": 0., "best_acc_hidx": 0}
    if args.load_from:
        restart_from_checkpoint(
            os.path.join(args.output_dir, args.load_from),
            models=linear_classifiers,
            optimizers=optimizers,
            schedulers=schedulers,
            run_variables=to_restore)
    start_epoch = to_restore["epoch"]
    best_acc = to_restore["best_acc"]
    best_acc_hidx = to_restore["best_acc_hidx"]


    for epoch in range(start_epoch, args.epochs):
        train_loader.sampler.set_epoch(epoch)
        model.eval()
        linear_classifiers.train()
        train_stats = train_one_epoch(
            model, linear_classifiers, optimizers, train_loader, epoch,
            args.n_last_blocks, args.avgpool_patchtokens, args.permutes)
        for scheduler in schedulers:
            scheduler.step()

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        if epoch % args.val_freq == 0 or epoch == args.epochs - 1:
            model.eval()
            linear_classifiers.eval()
            test_stats = validate_network(
                val_loader, model, linear_classifiers,
                args.n_last_blocks, args.avgpool_patchtokens, args.permutes)

            group_best_acc = 0
            group_best_acc_hidx = 0
            group_best_acc_sweep_lr_only = 0
            for group, pm in enumerate(args.permutes):
                lr, wd, optim = pm
                # print(f"Accuracy at epoch {epoch} with lr {lr:.5f} wd {wd:.0e} optim {optim:4} of the network \
                #         on the {len(dataset_val)} test images: {test_stats['acc{}@1'.format(group)]:.1f}%")
                if group % (len(args.wds) * len(args.optims)) == 0:
                    group_best_acc_sweep_lr_only = max(group_best_acc_sweep_lr_only, test_stats['acc{}@1'.format(group)])
                # group_best_acc = max(group_best_acc, test_stats['acc{}@1'.format(group)])
                if test_stats['acc{}@1'.format(group)] >= group_best_acc:
                    group_best_acc_hidx = group
                    group_best_acc = test_stats['acc{}@1'.format(group)]

            log_stats = {**{k: v for k, v in log_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()}}

            key = "teacher" if args.eval_teacher else "student"
            with (Path(args.output_dir) / f"log_{key}.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

            if group_best_acc > best_acc:
                best_acc = group_best_acc
                best_acc_hidx = group_best_acc_hidx
            print(f'Max accuracy so far: {best_acc:.2f}%')

            if misc.is_main_process() and (group_best_acc >= best_acc):
                save_dict = {
                    "epoch": epoch + 1,
                    "state_dict": linear_classifiers.state_dict(),
                    "optimizers": [optimizer.state_dict() for optimizer in optimizers],
                    "schedulers": [scheduler.state_dict() for scheduler in schedulers],
                    "best_acc": best_acc,
                    'best_acc_hidx': best_acc_hidx,
                }
                torch.save(save_dict, os.path.join(args.output_dir, f"checkpoint_{key}_linear.pth"))

            lr, wd, optim = args.permutes[best_acc_hidx]
            log_best_stats = {"epoch": epoch, "best_acc": best_acc, "best_lr": lr, "best_wd": wd, "best_optim": optim}
            if misc.is_main_process():
                with (Path(args.output_dir) / f"log_best_{key}.txt").open("a") as f:
                    f.write(json.dumps(log_best_stats) + "\n")

    lr, wd, optim = args.permutes[best_acc_hidx]
    print("Training of the supervised linear classifier on frozen features completed.\n",
              "Top-1 test accuracy: {acc:.1f}\n".format(acc=best_acc),
              "Optim configs with top-1 test accuracy: lr {lr:.5f}, wd {wd:.0e}, optim {optim:4}\n".format(lr=lr, wd=wd, optim=optim))


def train_one_epoch(model, linear_classifiers, optimizers, loader, epoch, n, avgpool, permutes):
    metric_logger = misc.MetricLogger(delimiter="  ")
    for group, _ in enumerate(permutes):
        metric_logger.add_meter('lr{}'.format(group), misc.SmoothedValue(window_size=1, fmt='{value:.5f}'))
    header = 'Epoch: [{}]'.format(epoch)
    for (inp, target) in metric_logger.log_every(loader, 100, header):
        # move to gpu
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # forward
        with torch.no_grad():
            intermediate_output = model.get_intermediate_layers(inp, n)
            if avgpool == 0:
                # norm(x[:, 0])
                output = [x[:, 0] for x in intermediate_output]
            elif avgpool == 1:
                # x[:, 1:].mean(1)
                output = [torch.mean(intermediate_output[-1][:, 1:], dim=1)]
            elif avgpool == 2:
                # norm(x[:, 0]) + norm(x[:, 1:]).mean(1)
                output = [x[:, 0] for x in intermediate_output] + [torch.mean(intermediate_output[-1][:, 1:], dim=1)]
            else:
                assert False, "Unkown avgpool type {}".format(avgpool)

            output = torch.cat(output, dim=-1)

        losses = []
        for linear_classifier, optimizer in zip(linear_classifiers, optimizers):

            pred = linear_classifier(output)

            # compute cross entropy loss
            loss = nn.CrossEntropyLoss()(pred, target)

            # compute the gradients
            optimizer.zero_grad()
            loss.backward()

            # step
            optimizer.step()

            losses.append(loss)

        # log
        torch.cuda.synchronize()
        for group, (loss, optimizer) in enumerate(zip(losses, optimizers)):
            metric_logger.update(**{'loss{}'.format(group): loss.item()})
            metric_logger.update(**{'lr{}'.format(group): optimizer.param_groups[0]["lr"]})
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def validate_network(val_loader, model, linear_classifiers, n, avgpool, permutes):
    linear_classifiers.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'
    for inp, target in metric_logger.log_every(val_loader, 100, header):
        # move to gpu
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # forward
        with torch.no_grad():
            intermediate_output = model.get_intermediate_layers(inp, n)
            if avgpool == 0:
                # norm(x[:, 0])
                output = [x[:, 0] for x in intermediate_output]
            elif avgpool == 1:
                # x[:, 1:].mean(1)
                output = [torch.mean(intermediate_output[-1][:, 1:], dim=1)]
            elif avgpool == 2:
                # norm(x[:, 0]) + norm(x[:, 1:]).mean(1)
                output = [x[:, 0] for x in intermediate_output] + [torch.mean(intermediate_output[-1][:, 1:], dim=1)]
            else:
                assert False, "Unkown avgpool type {}".format(avgpool)

            output = torch.cat(output, dim=-1)

        losses = []
        acc1s = []
        acc5s = []
        for group, linear_classifier in enumerate(linear_classifiers):

            pred = linear_classifier(output)
            loss = nn.CrossEntropyLoss()(pred, target)
            losses.append(loss)

            if linear_classifier.module.num_labels >= 5:
                acc1, acc5 = accuracy(pred, target, topk=(1, 5))
                acc1s.append(acc1)
                acc5s.append(acc5)
            else:
                acc1, = accuracy(pred, target, topk=(1,))
                acc1s.append(acc1)

            batch_size = inp.shape[0]
            metric_logger.update(**{'loss{}'.format(group): loss.item()})
            metric_logger.meters['acc{}@1'.format(group)].update(acc1.item(), n=batch_size)
            if linear_classifier.module.num_labels >= 5:
                metric_logger.meters['acc{}@5'.format(group)].update(acc5.item(), n=batch_size)

    for group, (pm, linear_classifier) in enumerate(zip(permutes, linear_classifiers)):
        lr, wd, optim = pm
        if linear_classifier.module.num_labels >= 5:
            print('* [Lr {lr:.5f} Wd {wd:.0e} Optim {optim:4}] Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
            .format(lr=lr, wd=wd, optim=optim,
                    top1=metric_logger.meters['acc{}@1'.format(group)],
                    top5=metric_logger.meters['acc{}@5'.format(group)],
                    losses=metric_logger.meters['loss{}'.format(group)]))
        else:
            print('* [Lr {lr:.5f} Wd {wd:.0e} Optim {optim:4}] Acc@1 {top1.global_avg:.3f} loss {losses.global_avg:.3f}'
            .format(lr=lr, wd=wd, optim=optim,
                    top1=metric_logger.meters['acc{}@1'.format(group)],
                    losses=metric_logger.meters['loss{}'.format(group)]))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


class LinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features"""
    def __init__(self, dim, num_labels=1000):
        super(LinearClassifier, self).__init__()
        self.num_labels = num_labels
        self.linear = nn.Linear(dim, num_labels)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
        # flatten
        x = x.view(x.size(0), -1)

        # linear layer
        return self.linear(x)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluation with linear classification on ImageNet')
    
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--seed', default=0, type=int)

    # Model parameters
    parser.add_argument('--arch', default='vit_base_patch16', type=str, choices=['vit_small_patch16', 'vit_base_patch16', 'vit_large_patch16'], help='Architecture.')
    parser.add_argument('--use_fc_norm', action='store_true')
    parser.add_argument('--pretrained_weights', default='', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument('--eval_teacher', action='store_true')
    parser.add_argument('--n_last_blocks', default=1, type=int, help="""Concatenate [CLS] tokens
        for the `n` last blocks. We use `n=4` when evaluating ViT-Small and `n=1` with ViT-Base/Large.""")
    parser.add_argument('--avgpool_patchtokens', default=1, choices=[0, 1, 2], type=int,
        help="""Whether to use the [CLS] token (option 0), the global average pooled features (option 1), or both (option 2).""")

    # Optimizer parameters
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument('--batch_size', default=128, type=int, help='Batch size per GPU')
    parser.add_argument('--sweep_lr_only', default=True, type=bool, help='Wether or not to only sweep over learning rate')

    # Dataset parameters
    parser.add_argument('--input_size', default=224, type=int, help='images input size')
    parser.add_argument('--data_path', default='/datasets_local/ImageNet/', type=str, help='dataset path')
    parser.add_argument('--subset', default=-1, type=int, help='The number of images per class that they would be use for '
                        'training (default -1). If -1, then all the availabe images are used.')                    
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--num_labels', default=1000, type=int, help='Number of labels for linear classifier')

    # Distributed training parameters
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--dist_on_itp', action='store_true')           
    parser.add_argument('--backend', default='nccl', type=str, help='Specify backend nccl or gloo')
    parser.add_argument('--dist_eval', action='store_true', default=False, 
        help='Enabling distributed evaluation (recommended during training for faster monitor')

    parser.add_argument('--val_freq', default=1, type=int, help="Epoch frequency for validation.")
    parser.add_argument('--output_dir', default=".", help='Path to save logs and checkpoints')
    parser.add_argument('--load_from', default=None, help='Path to load checkpoints to resume training')

    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    eval_linear(args)
