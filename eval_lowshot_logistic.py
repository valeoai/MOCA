"""
Adapted from the MSN[*] code: https://github.com/facebookresearch/msn
[*] Masked Siamese Networks for Label-Efficient Learning, Assran et al, ECCV 22.

NOTE: The initial MSN code and the code used for the TMLR submission both utilized the cyanure package 
to solve the logistic regression problem. However, when I was preparing the code for release, 
I encountered problems installing cyanure on the machines I had access to. As a result, 
I modified the code to use sklearn.linear_model for logistic regression. 
This change not only made installation easier but also resulted in faster performance.
"""

import os
import argparse
import logging
import datetime
import pprint

import numpy as np
from torch import nn
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import sklearn.linear_model

import numpy as np
from util.model_utils import interpolate_pos_embed

import models_vit
from util.model_utils import load_pretrained_encoder

from collections import OrderedDict
from pathlib import Path
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument(
    '--penalty', type=str,
    help='regularization for logistic classifier',
    default='l2',
    choices=[
        'l2',
        'elastic-net'
    ])
parser.add_argument(
    '--mask', type=float,
    default=0.0,
    help='regularization')
parser.add_argument(
    '--preload', action='store_true',
    help='whether to preload embs if possible')
parser.add_argument(
    '--model_name', type=str,
    help='model architecture')
parser.add_argument(
    '--pretrained', type=str,
    help='path to pretrained model',
    default='')
parser.add_argument(
    '--device', type=str,
    default='cuda:0',
    help='device to run script on')
parser.add_argument(
    '--normalize', type=bool,
    default=True,
    help='whether to standardize images before feeding to nework')
parser.add_argument(
    '--data_path', type=str,
    default='/datasets/',
    help='root directory to data')
parser.add_argument(
    '--subset_path', type=str,
    default=None,
    help='name of dataset to evaluate on')
parser.add_argument(
    '--use_fc_norm', action='store_true')
parser.add_argument(
    '--use_teacher', type=bool,
    default=True,
    help='whether to use the teacher encoder.')
parser.add_argument(
    '--global_pool', type=bool,
    default=True,
    help='whether to use the avg-pooled patch token as image representation.')
parser.add_argument(
    '--remove_norm', type=bool,
    default=True,
    help='whether to remove the last LN layer.')

parser.add_argument(
    '--use_sklearn', action='store_true',
        help='Use the sklearn.linear_model for solving the logistic regression problem')
parser.add_argument(
    '--use_cyanure', action='store_false', dest='use_sklearn')
parser.set_defaults(use_sklearn=True)    

parser.add_argument(
    '--lambd', type=float, nargs='+',
    default=[0.1,],
    help='regularization')

parser.add_argument(
    '--output_dir', type=str,
    help='Where to store stuff.',
    default='')

parser.add_argument('--nthreads', type=int, default=4, help='Number of threads.')


logging.basicConfig()
logger = logging.getLogger("LowShot")
strHandler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)-8s - %(levelname)-6s - %(message)s')
strHandler.setFormatter(formatter)
logger.addHandler(strHandler)
logger.setLevel(logging.INFO)

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

pp = pprint.PrettyPrinter(indent=4)


def init_data(
    transform,
    batch_size,
    pin_mem=False,
    num_workers=8,
    world_size=1,
    rank=0,
    data_path=None,
    training=True,
    drop_last=True,
    subset_file=None
):
    split = "train" if training else "val"
    data_path = os.path.join(data_path, split)
    dataset = torchvision.datasets.ImageFolder(root=data_path, transform=transform)
    if subset_file is not None:
        dataset = ImageNetSubset(dataset, subset_file)
    logger.info('ImageNet dataset created')
    if world_size == 1:
        dist_sampler = torch.utils.data.SequentialSampler(dataset)
    else:
        dist_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset=dataset,
            num_replicas=world_size,
            rank=rank)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers)

    return (data_loader, dist_sampler)


class ImageNetSubset(object):
    def __init__(self, dataset, subset_file):
        """
        ImageNetSubset
        :param dataset: ImageNet dataset object
        :param subset_file: '.txt' file containing IDs of IN1K images to keep
        """
        self.dataset = dataset
        self.subset_file = subset_file
        self.filter_dataset_(subset_file)

    def filter_dataset_(self, subset_file):
        """ Filter self.dataset to a subset """
        root = self.dataset.root
        class_to_idx = self.dataset.class_to_idx
        # -- update samples to subset of IN1k targets/samples
        new_samples = []
        logger.info(f'Using {subset_file}')
        with open(subset_file, 'r') as rfile:
            for line in rfile:
                class_name = line.split('_')[0]
                target = class_to_idx[class_name]
                img = line.split('\n')[0]
                new_samples.append(
                    (os.path.join(root, class_name, img), target)
                )
        self.samples = new_samples

    @property
    def classes(self):
        return self.dataset.classes

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        img = self.dataset.loader(path)
        if self.dataset.transform is not None:
            img = self.dataset.transform(img)
        if self.dataset.target_transform is not None:
            target = self.dataset.target_transform(target)
        return img, target


def main(
    blocks,
    lambd,
    mask_frac,
    preload,
    pretrained,
    subset_path,
    data_path,
    output_dir,
    penalty='l2',
    model_name=None,
    normalize=True,
    device_str='cuda:0', #?
    global_pool=1,
    use_fc_norm=False,
    use_teacher=True,
    remove_norm=True,
    nthreads=4,
    use_sklearn=True,
):
    device = torch.device(device_str)
    if 'cuda' in device_str:
        torch.cuda.set_device(device)


    # -- Define file names used to save computed embeddings (for efficient
    # -- reuse if running the script more than once)
    fname = "teacher" if use_teacher else "student"
    if use_fc_norm:
        fname += "_fc_norm"
    if global_pool:
        fname += "_avgpool"
    if remove_norm:
        fname += '_remove_norm'
    subset_tag = '-'.join(subset_path.split('/')).split('.txt')[0] if subset_path is not None else 'imagenet_subses1-100percent'

    now_str = datetime.datetime.now().__str__().replace(' ','_')
    now_str = now_str.replace(' ','_').replace('-','').replace(':','')
    lambd_str = str(lambd).replace('.','p')
    output_log_path = os.path.join(output_dir, f"log-{subset_tag}-{fname}-lambd{lambd_str}-{now_str}.txt")
    #output_log_path = os.path.join(output_dir, f"log-{subset_tag}-{fname}-multi_lambd-{now_str}.txt")
    logger.addHandler(logging.FileHandler(output_log_path))
    logger.info(f"Logging at: {output_log_path}")

    train_embs_path = os.path.join(output_dir, f'train-features-{subset_tag}-{fname}')
    test_embs_path = os.path.join(output_dir, f'val-features-{fname}')
    logger.info(f"Training features path: {train_embs_path}")
    logger.info(f"Testing features path: {test_embs_path}")

    # -- Function to make train/test dataloader
    def init_pipe(training):
        # -- make data transforms
        transform = transforms.Compose([
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225))])
        # -- init data-loaders/samplers
        subset_file = f"./eval/{subset_path}" if training else None
        data_loader, _ = init_data(
            transform=transform,
            batch_size=16,
            num_workers=nthreads,
            world_size=1,
            rank=0,
            data_path=data_path,
            training=training,
            drop_last=False,
            subset_file=subset_file)
        return data_loader

    # -- Initialize the model
    encoder = init_model(
        model_name=model_name,
        global_pool=global_pool,
        use_fc_norm=use_fc_norm,
        use_teacher=use_teacher,
        pretrained=pretrained,
        remove_norm=remove_norm,
        device=device)
    encoder.eval()

    # -- If train embeddings already computed, load file, otherwise, compute
    # -- embeddings and save
    if preload and os.path.exists(train_embs_path):
        checkpoint = torch.load(train_embs_path, map_location='cpu')
        embs, labs = checkpoint['embs'], checkpoint['labs']
        logger.info(f'loaded embs of shape {embs.shape}')
    else:
        data_loader = init_pipe(True)
        embs, labs = make_embeddings(
            blocks=blocks,
            device=device,
            mask_frac=mask_frac,
            data_loader=data_loader,
            encoder=encoder)
        torch.save({
            'embs': embs,
            'labs': labs
        }, train_embs_path)
        logger.info(f'saved train embs of shape {embs.shape}')

    # -- If test embeddings already computed, load file, otherwise, compute
    # -- embeddings and save
    if preload and os.path.exists(test_embs_path):
        checkpoint = torch.load(test_embs_path, map_location='cpu')
        test_embs, test_labs = checkpoint['embs'], checkpoint['labs']
        logger.info(f'loaded test embs of shape {test_embs.shape}')
    else:
        data_loader = init_pipe(False)
        test_embs, test_labs = make_embeddings(
            blocks=blocks,
            device=device,
            mask_frac=0.0,
            data_loader=data_loader,
            encoder=encoder)
        torch.save({
            'embs': test_embs,
            'labs': test_labs
        }, test_embs_path)
        logger.info(f'saved test embs of shape {test_embs.shape}')
    
    if use_sklearn:
        # Installing cyanure, which the original MSN code uses, is tricky in the machines that I have access to.
        # Instead I use sklearn, which is simpler to install much faster

        # -- Normalize embeddings
        embs = preprocess_embeddings(embs, normalize)
        test_embs = preprocess_embeddings(test_embs, normalize)

        for lambd_val in lambd:
            logger.info(f"Lambda value {lambd_val} --- C value {1.0 / lambd_val}")
            # -- Fit Logistic Regression Classifier
            classifier = sklearn.linear_model.LogisticRegression(
                penalty=penalty, dual=False, tol=0.0001, C=(1.0 / lambd_val),
                fit_intercept=False, intercept_scaling=1, solver='lbfgs',
                max_iter=300, multi_class='multinomial', verbose=0, warm_start=False, n_jobs=nthreads)
            classifier = classifier.fit(embs.numpy(), labs.numpy())

            # -- Evaluate and log
            train_score = classifier.score(embs.numpy(), labs.numpy())
            logger.info(f'train score (lambd_val {lambd_val}): {train_score}')

            # -- Evaluate and log
            test_score = classifier.score(test_embs.numpy(), test_labs.numpy())
            logger.info(f'test score (lambd_val {lambd_val}): {test_score}\n\n')
    else:
        import cyanure as cyan
        # -- Normalize embeddings
        cyan.preprocess(embs, normalize=normalize, columns=False, centering=True)
        cyan.preprocess(test_embs, normalize=normalize, columns=False, centering=True)

        for lambd_val in lambd:
            logger.info(f"Lambda value {lambd_val}")
            # -- Fit Logistic Regression Classifier
            classifier = cyan.MultiClassifier(loss='multiclass-logistic', penalty=penalty, fit_intercept=False)
            classifier.fit(
                embs.numpy(), labs.numpy(), it0=10, lambd=lambd, lambd2=lambd,
                nthreads=nthreads, tol=1e-3, solver='auto', seed=0, max_epochs=300)

            # -- Evaluate and log
            train_score = classifier.score(embs.numpy(), labs.numpy())
            logger.info(f'train score (lambd_val {lambd_val}): {train_score}')
            
            # -- Evaluate and log
            test_score = classifier.score(test_embs.numpy(), test_labs.numpy())
            logger.info(f'test score (lambd_val {lambd_val}): {test_score}\n\n')

    return test_score


def preprocess_embeddings(embs, normalize):
    # Equivalant to cyan.preprocess(embs, normalize=normalize, columns=False, centering=True)
    embs -= embs.mean(dim=1, keepdim=True) # Centering
    if normalize: # L2-normalization
        embs = F.normalize(embs, p=2.0, dim=1)
    return embs


def make_embeddings(
    blocks,
    device,
    mask_frac,
    data_loader,
    encoder,
):
    ipe = len(data_loader)

    z_mem, l_mem = [], []

    for itr, (imgs, labels) in enumerate(data_loader):
        imgs = imgs.to(device)
        with torch.no_grad():
            z = encoder.forward_blocks(imgs, blocks, mask_frac).cpu()
        labels = labels.cpu()
        z_mem.append(z)
        l_mem.append(labels)
        if itr % 50 == 0:
            logger.info(f'[{itr}/{ipe}]')

    z_mem = torch.cat(z_mem, 0)
    l_mem = torch.cat(l_mem, 0)
    logger.info(z_mem.shape)
    logger.info(l_mem.shape)

    return z_mem, l_mem


def init_model(
    model_name,
    global_pool,
    use_fc_norm,
    use_teacher,
    pretrained,
    remove_norm,
    device):
    # ============ building network ... ============
    model = models_vit.__dict__[model_name](
        num_classes=0,
        drop_path_rate=0,
        global_pool=global_pool,
        fc_norm=use_fc_norm)

    print(f"Model {model_name} built.")
    # load weights to evaluate
    if pretrained:
        load_pretrained_encoder(model, pretrained, 
            use_teacher=use_teacher, avg_pooling=global_pool,
            use_fc_norm=use_fc_norm)

    if remove_norm:
        model.norm = nn.Identity()

    model.to(device)
    return model


if __name__ == '__main__':
    """'main' for launching script using params read from command line"""
    global args
    args = parser.parse_args()
    pp.pprint(args)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    torch.multiprocessing.set_sharing_strategy('file_system')
    main(
        blocks=1,
        lambd=args.lambd,
        penalty=args.penalty,
        mask_frac=args.mask,
        preload=args.preload,
        pretrained=args.pretrained,
        subset_path=args.subset_path,
        data_path=args.data_path,
        model_name=args.model_name,
        normalize=args.normalize,
        device_str=args.device,
        use_fc_norm=args.use_fc_norm,
        use_teacher=args.use_teacher,
        global_pool=args.global_pool,
        remove_norm=args.remove_norm,
        output_dir=args.output_dir,
        nthreads=args.nthreads,
        use_sklearn=args.use_sklearn)
