# Copyright 2024 - Valeo Comfort and Driving Assistance - valeo.ai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import datetime
import json
import numpy as np
import os
import time
import math
import sys
from pathlib import Path
from typing import Iterable

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import models_moca

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.transforms import DataAugmentationTwoViews
from util.data_utils import subset_of_ImageNet_train_split, prepare_knn_imagenet_datasets
from util.knn_utils import knn_evaluation_pipeline


def get_args_parser():
    parser = argparse.ArgumentParser('MOCA pre-training', add_help=False)
    parser.add_argument('--data_path', default='/datasets_local/ImageNet/', type=str, help='dataset path')
    parser.add_argument('--output_dir', default='./output_dir', help='path where to save, empty for no saving')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--resume_if_exists', action='store_true')
    parser.add_argument('--resume_only_model', action='store_true')

    # ViT parameters
    parser.add_argument('--model', default='moca_vit_base_patch16_dec2', type=str, metavar='MODEL', help='Name of model to train')
    parser.add_argument('--input_size', default=224, type=int, help='images input size')
    
    # MOCA parameters:
    parser.add_argument('--kappa', default=3.0, type=float, 
        help="The inverse temperature value that is used in the prediction task.")
    parser.add_argument('--inv_delta', default=10.0, type=float, 
        help="The inverse temperature value is used for the soft-assigment of the patch tokens to the visual words.")
    parser.add_argument('--num_words', default=4096, type=int, help="Vocabulary size.") # 5
    parser.add_argument('--img_weight', default=0.5, type=float, help="Weight of the BoW prediciton loss.")
    parser.add_argument('--loc_weight', default=0.5, type=float, help="Weight of the MoW prediction loss (i.e., the MIM loss or dense loss).")
    parser.add_argument('--num_new_words', default=32, type=int, help="How many new visual words will be added to the queue-based vocabulary at each training iteration.")
    parser.add_argument('--early_layer', type=int, default=4)
    parser.add_argument('--pretrained_teacher', default=None, type=str, help="Use a pretrained teacher.")
    parser.add_argument('--pred_mlp_ratio', default=2.0, type=float, help="Multiplier of the number of channels in the hidden layer of the weight generation modules for the BoW and MoW prediction tasks.")
    parser.add_argument('--avg_pooling', action='store_true', help="Use the avg feature of the patch tokens for the prediction tasks (instead of the [CLS] token feature)")
    parser.add_argument('--no_avg_pooling', action='store_false', dest='avg_pooling')
    parser.set_defaults(avg_pooling=True)    
    parser.add_argument('--mask_ratio', default=[0.55, 0.75], type=float, nargs='+',
        help='Masking ratio (percentage of removed patches).')
    parser.add_argument('--dec_mask_ratio', default=[0.35, 0.55], type=float, nargs='+',
        help='Masking ratio (percentage of removed patches) for the decoder. It must be smaller than --mask_ratio.')

    # Optimizer parameters
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--batch_size', default=64, type=int,
        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--accum_iter', default=1, type=int,
        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    
    # Optimizer --- learning rate schedule
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--max_epochs', default=200, type=int, help="The number of epochs for which the training will last.")
    parser.add_argument('--lr_epochs_mul', default=1.25, type=float,
        help="The cosine lr schedule decreases the lr rate from the peak value to zero at epoch int(lr_epochs_mul * max_epochs). "
             "By using a lr_epochs_mul > 1, we can stop the training before the lr reaches 0. "
             "lr_epochs_mul must be >= 1")
    parser.add_argument('--warmup_epochs', type=int, default=30, metavar='N', help='epochs to warmup LR')
    parser.add_argument('--lr', type=float, default=None, metavar='LR', help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1.5e-4, metavar='LR', help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--lr_wgen', type=float, default=1e-4, metavar='LR', help='learning rate (absolute lr) for the weight generation modules. This learning rate remains constant during training.')

    # Optimizer --- weight decay
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--weight_decay_end', type=float, default=None,
                        help='Final weight decay value (default: None). If weight_decay_end != None, then an annealing cosine schedule will be used. It has been shown that this helps ViTs.')
    
    # Optimizer --- teacher momentum parameters:
    parser.add_argument('--alpha', default=0.994, type=float, help="Momentum value for the teacher encoder.") # 1 momentum
    parser.add_argument('--alpha_cos', action='store_true', help="Use cosine annealing for the teacher momentum alpha")
    parser.add_argument('--alpha_constant', action='store_false', dest='alpha_cos', help="Use constant valude for the teacher momentum alpha")
    parser.set_defaults(alpha_cos=True)       

    # Optimizer --- clip grad value / fp16 
    parser.add_argument('--clip_grad', type=float, default=0.1,
        help="""Maximal parameter gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can help optimization for larger ViT architectures. 0 for disabling.""")
    parser.add_argument('--no_fp16', action='store_true')
    parser.add_argument('--init_scale', type=float, default=65536.0)

    # Data loading parameters
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--crop_min', default=0.2, type=float, 
        help='minimum scale for random cropping (default: 0.2)')    
    parser.add_argument('--subset', default=-1, type=int,
        help='The number of images per class that they would be use for '
             'training (default -1). If -1, then all the availabe images are used. Usefull for debugging.')
    parser.add_argument('--pin_mem', action='store_true',
        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--backend', default='nccl', type=str, help='Specify backend nccl or gloo')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')

    # K-nn evaluation specific arguments.
    parser.add_argument('--eval_every', default=20, type=int, help='How frequently to run evaluation (epochs)')
    parser.add_argument('--start_with_knn', action='store_true')
    parser.add_argument('--skip_knn', action='store_true')
    parser.add_argument('--evaluate', action='store_true', help="It only evaluates the resumed model.")
    parser.add_argument('--nb_knn', default=[10, 20,], nargs='+', type=int,
                        help='Number of NN to use. 20 is usually working the best.')
    parser.add_argument('--temperature', default=0.07, nargs='+', type=float,
                        help='Temperature used in the voting coefficient')
    parser.add_argument('--eval_teacher', action='store_true')
    parser.add_argument('--num_workers_knn', default=None, type=int)
    parser.add_argument('--batch_size_knn', default=None, type=int)

    # Misceleneous.
    parser.add_argument('--set_detect_anomaly', action='store_true')
    parser.add_argument('--log_freq', default=20, type=int)
    parser.add_argument('--no_tensorboard', action='store_true')

    return parser


def initialize_optimizer(model, args):
    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    print(f"accumulate grad iterations: {args.accum_iter}")
    print(f"effective batch size: {eff_batch_size}")

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256
    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    if args.lr_wgen is None:
        args.lr_wgen = args.lr
    args.weight_decay_wgen = args.weight_decay
    print("WGEN modules:")
    print("==> actual lr_wgen: %.2e" % args.lr_wgen)
    print("==> weight_decay_wgen: %.2e" % args.weight_decay_wgen)

    param_groups_dict = {"no_decay": [], "decay": [], "wgen_no_decay": [], "wgen_decay": []}
    wgen_prefixes = ("decoder_pred.layers_w.", "encoder_pred.layers_w.")
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # set wd as 0 for bias and norm layers
        key = "no_decay" if (param.ndim <= 1 or name.endswith(".bias")) else "decay"
        if any(name.startswith(prefix) for prefix in wgen_prefixes):
            key = "wgen_" + key
        param_groups_dict[key].append(param)

    for key, params in param_groups_dict.items():
        print(f"Param group {key} size {len(params)}")

    param_groups = [
        {'params': param_groups_dict["no_decay"],      'is_wgen': False, 'use_wd': False, 'weight_decay': 0.},
        {'params': param_groups_dict["decay"],         'is_wgen': False, 'use_wd': True,  'weight_decay': args.weight_decay},
        {'params': param_groups_dict["wgen_no_decay"], 'is_wgen': True,  'use_wd': False, 'weight_decay': 0.},
        {'params': param_groups_dict["wgen_decay"],    'is_wgen': True,  'use_wd': True,  'weight_decay': args.weight_decay_wgen},
    ]
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    print(optimizer)

    loss_scaler = None if args.no_fp16 else NativeScaler(init_scale=args.init_scale)

    return optimizer, loss_scaler, args


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device,
                    epoch: int,
                    loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('optim/lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('optim/m',  misc.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss/tot', misc.SmoothedValue(window_size=1, fmt='{value:.2f} ({global_avg:.2f})'))
    metric_logger.add_meter('loss/img', misc.SmoothedValue(window_size=1, fmt='{value:.2f} ({global_avg:.2f})'))
    metric_logger.add_meter('loss/loc', misc.SmoothedValue(window_size=1, fmt='{value:.2f} ({global_avg:.2f})'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20
    backward_fun = loss_scaler if (loss_scaler is not None) else misc.backward

    optimizer.zero_grad()

    for data_iter_step, (samples, _) in enumerate(metric_logger.log_every(data_loader, print_freq * args.accum_iter, header)):
        update_grad = (data_iter_step+1) % args.accum_iter == 0
        update_teacher = True if (args.accum_iter==1) else update_grad
        log_stats = (args.log_freq > 0) and (((data_iter_step+1)//args.accum_iter - 1) % args.log_freq == 0)
        # we use a per iteration (instead of per epoch) lr scheduler
        epoch_float = data_iter_step / len(data_loader) + epoch
        if data_iter_step % args.accum_iter == 0:
            misc.adjust_learning_rate(optimizer, epoch_float, args.lr, args.lr_epochs, args.warmup_epochs, args.lr_wgen)
            misc.adjust_weight_decay(optimizer, epoch_float, args.max_epochs, args.weight_decay, args.weight_decay_end)
            momentum = misc.adjust_encoder_momentum(args.alpha, epoch_float, args.max_epochs, args.alpha_cos, alpha_max=1.0)

        x1 = samples[0].cuda(args.gpu, non_blocking=True)
        x2 = samples[1].cuda(args.gpu, non_blocking=True)
        with torch.cuda.amp.autocast(loss_scaler is not None):
            loss, stats = model(x1, x2, momentum=momentum, args=args, update_teacher=update_teacher)

        loss_value = loss.item()
        losses = stats["losses"]
        if not math.isfinite(loss_value):
            print(f"Rank {args.rank} - Loss is {loss_value} ({losses}), stopping training", force=True)
            print(f"Rank {args.rank} - Stats {stats}", force=True)
            sys.exit(1)

        loss /= args.accum_iter
        backward_fun(
            loss, optimizer, parameters=model.parameters(),
            update_grad=update_grad, clip_grad=args.clip_grad)

        if update_grad:
            optimizer.zero_grad()
        torch.cuda.synchronize()

        metric_logger.update(**{"loss/tot": loss_value, "loss/img": losses["img"], "loss/loc": losses["loc"]})
        metric_logger.update(**{"optim/lr": optimizer.param_groups[0]["lr"], "optim/m": momentum})

        if (log_stats and (log_writer is not None) and update_grad):
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int(epoch_float * 1000)
            log_writer.add_scalar('train_losses/total', loss_value, epoch_1000x)
            log_writer.add_scalar('train_losses/bow', losses["img"], epoch_1000x)
            log_writer.add_scalar('train_losses/dense', losses["loc"], epoch_1000x)

            scale = loss_scaler.get_scale() if loss_scaler is not None else 1.0
            log_writer.add_scalar('train_optim/lr', optimizer.param_groups[0]["lr"], epoch_1000x)
            log_writer.add_scalar('train_optim/wd', optimizer.param_groups[1]["weight_decay"], epoch_1000x)
            log_writer.add_scalar('train_optim/momentum', momentum, epoch_1000x)
            log_writer.add_scalar('train_optim/lrwgen', optimizer.param_groups[2]["lr"], epoch_1000x)

    # Gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))
    device = torch.device(args.gpu)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True
    transform_train = DataAugmentationTwoViews(input_size=args.input_size, crop_min=args.crop_min)
    dataset_train = datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=transform_train)
    print(dataset_train)
    if (args.subset is not None) and (args.subset >= 1):
        dataset_train = subset_of_ImageNet_train_split(dataset_train, args.subset)

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)
    print("Sampler_train = %s" % str(sampler_train))

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True)
        
    if args.num_workers_knn is None:
        args.num_workers_knn = args.num_workers
    if args.batch_size_knn is None:
        args.batch_size_knn = args.batch_size
    dataset_train_knn, data_loader_train_knn, dataset_val_knn, data_loader_val_knn = prepare_knn_imagenet_datasets(
        args.data_path, global_rank, num_tasks, input_size=args.input_size, subset=args.subset,
        batch_size=args.batch_size_knn, num_workers=args.num_workers_knn, pin_mem=args.pin_mem)

    print(f"Data loaded for training: there are {len(dataset_train)} images.")
    print(f"Data loaded for K-NN training: there are {len(dataset_train_knn)} images.")
    print(f"Data loaded for K-NN validation: there are {len(dataset_val_knn)} images.")

    if global_rank == 0 and (not args.no_tensorboard): #args.log_dir is not None:
        args.log_dir = os.path.join(args.output_dir, "tensorboard")
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    assert isinstance(args.mask_ratio, list)
    assert isinstance(args.dec_mask_ratio, list)
    args.num_mviews = len(args.mask_ratio)
    #assert len(args.mask_ratio) == args.num_mviews
    assert len(args.dec_mask_ratio) == args.num_mviews
    assert all([isinstance(m, float) for m in args.mask_ratio])
    assert all([isinstance(m, float) for m in args.dec_mask_ratio])
    print(f"Masking ratio(s): {args.mask_ratio}")
    print(f"Decoder masking ratio(s): {args.dec_mask_ratio}")

    # define the model
    model = models_moca.__dict__[args.model](
        kappa=args.kappa, inv_delta=args.inv_delta, num_words=args.num_words,
        num_new_words=args.num_new_words, pred_mlp_ratio=args.pred_mlp_ratio,
        early_layer=args.early_layer, use_loc_loss=args.loc_weight>0.0)

    model.to(args.gpu)
    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
    model_without_ddp = model.module

    optimizer, loss_scaler, _ = initialize_optimizer(model_without_ddp, args)
    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.max_epochs} epochs")
    args.lr_epochs = int(args.max_epochs * args.lr_epochs_mul) # set the learning rate schedule so that it reaches zero at 1.25 * max_epochs
    print(f"==> Learning rate schedule based on {args.lr_epochs} epochs")
    print(f"==> Learning rates lr: {args.lr} and lr_wgen: {args.lr_wgen}")

    encoder_knn = model_without_ddp.encoder_teacher if args.eval_teacher else model_without_ddp.encoder
    if args.evaluate or (args.start_with_knn and args.start_epoch > 0):
        knn_results = knn_evaluation_pipeline(
            encoder_knn, dataset_train_knn, data_loader_train_knn, dataset_val_knn, data_loader_val_knn,
            avg_pooling=args.avg_pooling, temperature=args.temperature, nb_knn=args.nb_knn)

    if args.evaluate:
        return

    args.output_log_file = os.path.join(args.output_dir, "log.txt")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.max_epochs):
        data_loader_train.sampler.set_epoch(epoch)
        if log_writer is not None:
            print('log_dir: {}'.format(log_writer.log_dir))

        train_stats = train_one_epoch(
            model, data_loader_train, optimizer, device, epoch, loss_scaler,
            log_writer=log_writer, args=args)

        if args.output_dir and ((epoch % 10 == 0 and epoch > 0) or epoch + 1 == args.max_epochs):
            # Save a checkpoint every 10 epochs
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp,
                optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch)
        misc.save_model(
            args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
            loss_scaler=loss_scaler, epoch=epoch, filename="checkpoint-last")

        torch.cuda.synchronize()

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch,}
        if log_writer is not None:
            for k, v in train_stats.items():
                log_writer.add_scalar(f'Epoch_train_{k}', v, epoch)

        # Evaluation with k-NN
        if (epoch % args.eval_every == 0 or epoch == args.max_epochs - 1) and (args.skip_knn is False):
            knn_results = knn_evaluation_pipeline(
                encoder_knn, dataset_train_knn, data_loader_train_knn, dataset_val_knn, data_loader_val_knn,
                avg_pooling=args.avg_pooling, temperature=args.temperature, nb_knn=args.nb_knn)
            log_stats.update(knn_results)
            if log_writer is not None:
                for key, val in knn_results['k-NN'].items():
                    log_writer.add_scalar(f'Epoch_eval_top1_{key}', val['top1'], epoch)

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(args.output_log_file, mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.set_detect_anomaly:
        torch.autograd.set_detect_anomaly(True)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
