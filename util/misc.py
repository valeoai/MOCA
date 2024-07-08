"""
# References:
# MAE: https://github.com/facebookresearch/mae
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
"""

import builtins
import datetime
import os
import time
from collections import defaultdict, deque, OrderedDict
from pathlib import Path

import math
import torch
import torch.distributed as dist
from socket import gethostname


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        #force = force or (get_world_size() > 8)
        if is_master or force:
            now = datetime.datetime.now().time()
            builtin_print('[{}] '.format(now), end='')  # print with time stamp
            builtin_print(*args, **kwargs)

    builtins.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


@torch.no_grad()
def concat_all_gather(tensor):
    if get_world_size() > 1:
        tensors_gather = [torch.ones_like(tensor)
            for _ in range(dist.get_world_size())]
        dist.all_gather(tensors_gather, tensor, async_op=False)
        return torch.cat(tensors_gather, dim=0)
    else:
        return tensor


@torch.no_grad()
def reduce_all(tensor):
    if get_world_size() > 1:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def all_reduce_mean(x):
    world_size = get_world_size()
    if world_size > 1:
        x_reduce = torch.tensor(x).cuda()
        dist.all_reduce(x_reduce)
        x_reduce /= world_size
        return x_reduce.item()
    else:
        return x


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        print("Distributed init versio 2")
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
        args.world_size = int(os.environ["WORLD_SIZE"])
        gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])
        assert gpus_per_node == torch.cuda.device_count()
        max_num_workers = int(os.environ["SLURM_CPUS_PER_TASK"])
        args.node     = args.rank // gpus_per_node
        args.hostname = gethostname()
        print(f"SLURM Dist init: rank {args.rank}/{args.world_size} on {args.hostname} (node {args.node}) with {gpus_per_node}GPUs per node and {max_num_workers}cpus per gpu", flush=True)

        args.dist_backend = 'nccl'
        args.distributed = True
        torch.cuda.set_device(args.gpu)
        torch.distributed.init_process_group(backend=args.dist_backend, rank=args.rank, world_size=args.world_size)
        args.rank = torch.distributed.get_rank()
        args.gpu = args.rank % torch.cuda.device_count()
        args.node = args.rank // torch.cuda.device_count()
        args.world_size = torch.distributed.get_world_size()
        print(f"DDP: rank={args.rank} - gpu={args.gpu} - node={args.node} - world_size={args.world_size}", flush=True)
        torch.distributed.barrier()
        setup_for_distributed(args.rank==0)
        return
    else:
        print('Not using distributed mode')
        setup_for_distributed(is_master=True)  # hack
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}, gpu {}'.format(
        args.rank, args.dist_url, args.gpu), flush=True)

    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self, init_scale=65536.0,):
        self._scaler = torch.cuda.amp.GradScaler(init_scale=init_scale)

    def __call__(
        self,
        loss,
        optimizer,
        clip_grad=None,
        parameters=None,
        create_graph=False,
        update_grad=True,
    ):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
            self._scaler.step(optimizer)
            self._scaler.update()

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)

    def get_scale(self):
        return self._scaler.get_scale()


def backward(loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
    loss.backward(create_graph=create_graph)
    if update_grad:
        if clip_grad is not None:
            assert parameters is not None
            norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
        optimizer.step()


def save_model(args, epoch, model, model_without_ddp, optimizer, loss_scaler, filename=None):
    output_dir = Path(args.output_dir)
    epoch_name = str(epoch)
    if filename is None:
        checkpoint_paths = [output_dir / ('checkpoint-%s.pth' % epoch_name)]
    else:
        checkpoint_paths = [output_dir / f'{filename}.pth']
    for checkpoint_path in checkpoint_paths:
        to_save = {
            'model': model_without_ddp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'args': args,
        }
        if loss_scaler is not None:
            to_save['scaler'] = loss_scaler.state_dict()
        save_on_master(to_save, checkpoint_path)


def load_model(args, model_without_ddp, optimizer, loss_scaler):
    if args.resume:
        resume_if_exists = args.resume_if_exists if hasattr(args, 'resume_if_exists') else False
        resume_only_model = args.resume_only_model if hasattr(args, 'resume_only_model') else False
        eval = args.eval if hasattr(args, 'eval') else False

        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            if resume_if_exists and os.path.isfile(args.resume) is False:
                print(f"WARNING: File {args.resume} does not exist. Will start training from epoch 0.")
                return
            else:
                checkpoint = torch.load(args.resume, map_location='cpu')

        model_without_ddp.load_state_dict(checkpoint['model'])
        print("Resume checkpoint %s" % args.resume)
        if (('optimizer' in checkpoint) and
            ('epoch' in checkpoint) and
            (eval is False) and
            (resume_only_model is False)):
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
            print("With optim & sched!")


def create_random_mask(N, L, mask_ratio, device):
    """Creates a random mask for masking a sequence of patch tokens.

    In order to mask a sequence of patch tokes, we first shuffle the patch 
    tokens (using ids_shuffle) and then keep only the first len_keep = L * (1 - mask_ratio)
    patch tokens. So, a random mask is represented by the shuffling indices (ids_shuffle)
    and the number of tokens that will be kept (len_keep).
    """
    len_keep = int(L * (1 - mask_ratio))

    noise = torch.rand(N, L, device=device)  # noise in [0, 1]
    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove

    return ids_shuffle, len_keep


def create_random_mask_mult_rounds(N, L, mask_ratio_list, device):
    ids_shuffle_list, len_keep_list = [], []
    for mask_ratio in mask_ratio_list:
        ids_shuffle, len_keep = create_random_mask(N, L, mask_ratio, device)

        ids_shuffle_list.append(ids_shuffle)
        len_keep_list.append(len_keep)

    return ids_shuffle_list, len_keep_list


def make_two_complementary_masks(N, L, mask_ratio_list, dec_mask_ratio_list, device):
    assert len(mask_ratio_list) == 2
    assert len(dec_mask_ratio_list) == 2

    # Number of tokens visible by encoder in 1st masking round
    len_keep_v1 = int(L * (1 - mask_ratio_list[0]))
    # Number of tokens visible by decoder in 1st masking round
    len_dec_v1 = int(L * (1 - dec_mask_ratio_list[0]))
    # Number of tokens visible by encoder in 2nd masking round
    len_keep_v2 = int(L * (1 - mask_ratio_list[1]))
    # Number of tokens visible by decoder in 2nd masking round
    len_dec_v2 = int(L * (1 - dec_mask_ratio_list[1]))
    #assert (len_dec_v2 - len_keep_v2) == (len_dec_v1 - len_keep_v1)

    ids_shuffle_v1, _ = create_random_mask(N, L, mask_ratio_list[0], device)

    # tokens visible by encoder (1st masking round)
    ids_enc_v1 = ids_shuffle_v1[:,:len_keep_v1]
    # Aditional tokens visible by decoder (1st masking round)
    ids_dec_v1 = ids_shuffle_v1[:,len_keep_v1:len_dec_v1] 
    ids_rest_v1 = ids_shuffle_v1[:,len_dec_v1:] 

    # For creating ids_shuffle for the 2nd masking take
    assert ids_rest_v1.shape[1] >= len_keep_v2
    # Tokens visible by encoder (2nd masking round)
    ids_enc_v2 = ids_rest_v1[:,:len_keep_v2]
    # Aditional tokens visible by decoder in 2nd masking round. They are the same as in 1st masking round.
    ids_dec_v2 = ids_dec_v1
    
    ids_shuffle_v2 = torch.cat((ids_enc_v2, ids_dec_v2, ids_rest_v1[:,len_keep_v2:], ids_enc_v1), dim=1)

    assert ids_shuffle_v2.shape == ids_shuffle_v1.shape

    return [ids_shuffle_v1, ids_shuffle_v2], [len_keep_v1, len_keep_v2]


def mask_input(x, ids_shuffle, len_keep):
    """
    Mask the input x by essentially suffling x (according to ids_shuffle)
    and then keep the first len_keep patch tokens.
    x: [N, L, D], sequence
    """
    N, L, D = x.shape  # batch, length, dim

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

    return x_masked


def adjust_encoder_momentum(alpha, epoch, num_epochs, cosine=True, alpha_max=1.0):
    """Adjust momentum based on current epoch"""
    if cosine:
        if epoch <= num_epochs:
            return half_cycle_cosine_schedule(alpha_max, alpha, epoch, num_epochs)
        else:
            return alpha_max
    else:
        return alpha


def half_cycle_cosine_schedule(end_value, start_value, epoch, num_epochs):
    return end_value - 0.5 * (1. + math.cos(math.pi * epoch / num_epochs)) * (end_value - start_value)


def warmup_schedule(end_value, start_value, epoch, warmup_epochs):
    if (epoch < warmup_epochs) and (end_value != start_value):
        return start_value + (end_value - start_value) * epoch / warmup_epochs
    else:
        return end_value


def compute_lr(lr, min_lr, num_epochs, warmup_epochs, epoch, constant=False):
    if epoch < warmup_epochs:
        return warmup_schedule(end_value=lr, start_value=0.0, epoch=epoch, warmup_epochs=warmup_epochs)
    elif constant:
        return lr
    elif epoch > num_epochs:
        return min_lr
    else:
        return half_cycle_cosine_schedule(end_value=min_lr, start_value=lr, epoch=epoch-warmup_epochs, num_epochs=num_epochs-warmup_epochs)


def adjust_learning_rate(optimizer, epoch, lr, lr_epochs, warmup_epochs, lr_wgen):
    lr_model = compute_lr(lr, 0, lr_epochs, warmup_epochs, epoch, constant=False)
    for param_group in optimizer.param_groups:
        if param_group.get("is_wgen", False):
            param_group["lr"] = lr_wgen
        else:
            param_group["lr"] = lr_model


def adjust_weight_decay(optimizer, epoch, max_epochs, weight_decay, weight_decay_end=None):
    if weight_decay_end is not None:
        weight_decay_model = half_cycle_cosine_schedule(
            end_value=weight_decay_end, start_value=weight_decay,
            epoch=epoch, num_epochs=max_epochs)
        for param_group in optimizer.param_groups:
            if (param_group['use_wd'] and param_group.get("is_wgen", False) == False):
                param_group['weight_decay'] = weight_decay_model   
