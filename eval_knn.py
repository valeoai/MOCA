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
import numpy as np
import os
import torch
import torch.backends.cudnn as cudnn
import util.misc as misc
import models_vit

from util.model_utils import load_pretrained_encoder
from util.data_utils import prepare_knn_imagenet_datasets
from util.knn_utils import knn_evaluation_pipeline


def get_args_parser():
    parser = argparse.ArgumentParser('K-NN evaluation', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size per GPU')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--num_workers', default=4, type=int)

    # Model parameters
    parser.add_argument('--arch', default='vit_base_patch16', type=str, choices=['vit_small_patch16', 'vit_base_patch16', 'vit_large_patch16'], help='Architecture.')
    parser.add_argument('--use_fc_norm', action='store_true')
    parser.add_argument('--avg_pooling', action='store_true', help="Use the avg feature of the patch tokens for the prediction tasks (instead of the [CLS] token feature)")
    parser.add_argument('--no_avg_pooling', action='store_false', dest='avg_pooling')
    parser.set_defaults(avg_pooling=True)    
    parser.add_argument('--pretrained_weights', default='', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument('--eval_teacher', action='store_true')

    # Dataset parameters
    parser.add_argument('--input_size', default=224, type=int, help='images input size')
    parser.add_argument('--data_path', default='/datasets_local/ImageNet/', type=str, help='dataset path')
    parser.add_argument('--subset', default=-1, type=int, help='The number of images per class that they would be use for '
                        'training (default -1). If -1, then all the availabe images are used.')                    
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

    # K-nn evaluation specific arguments.
    parser.add_argument('--nb_knn', default=[10, 20,], nargs='+', type=int,
                        help='Number of NN to use. 20 is usually working the best.')
    parser.add_argument('--temperature', default=[0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10, 0.12, 0.15, 0.20], nargs='+', type=float,
                        help='Temperature used in the voting coefficient')
    
    return parser


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))
    device = torch.device(args.gpu)

    # fix the seed for reproducibility
    seed = misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    dataset_train_knn, data_loader_train_knn, dataset_val_knn, data_loader_val_knn = prepare_knn_imagenet_datasets(
        args.data_path, misc.get_rank(), misc.get_world_size(), input_size=args.input_size, subset=args.subset,
        batch_size=args.batch_size, num_workers=args.num_workers, pin_mem=args.pin_mem)

    print(f"Data loaded for K-NN training: there are {len(dataset_train_knn)} images.")
    print(f"Data loaded for K-NN validation: there are {len(dataset_val_knn)} images.")

    # define the model
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
    print("Model = %s" % str(model))

    knn_results = knn_evaluation_pipeline(
        model, dataset_train_knn, data_loader_train_knn, dataset_val_knn, data_loader_val_knn,
        avg_pooling=args.avg_pooling, temperature=args.temperature, nb_knn=args.nb_knn)


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)
