"""
Adapted from the DINO[*] code: https://github.com/facebookresearch/dino
[*] Emerging Properties in Self-Supervised Vision Transformers, ICCV'22
"""

import torch
import torch.nn as nn
import torch.distributed as dist
import util.misc as misc


@torch.no_grad()
def extract_features(encoder, data_loader, avg_pooling, use_cuda=True):
    metric_logger = misc.MetricLogger(delimiter="  ")
    features = None

    for samples, index in metric_logger.log_every(data_loader, 10):
        samples = samples.cuda(non_blocking=True)
        index = index.cuda(non_blocking=True)
        feats = encoder.forward_last_features(samples)
        # Compute image-wise feature
        if avg_pooling:  
            feats = feats[:,1:,:].mean(1) # Compute average patch token
        else: 
            feats = feats[:, 0, :] # Keep only cls token
        feats = feats.contiguous()

        # init storage feature matrix
        if misc.get_rank() == 0 and features is None:
            features = torch.zeros(len(data_loader.dataset), feats.shape[-1])
            if use_cuda:
                features = features.cuda(non_blocking=True)
            print(f"Storing features into tensor of shape {features.shape}")

        # get indexes from all processes
        y_all = torch.empty(misc.get_world_size(), index.size(0), dtype=index.dtype, device=index.device)
        y_l = list(y_all.unbind(0))
        y_all_reduce = torch.distributed.all_gather(y_l, index, async_op=True)
        y_all_reduce.wait()
        index_all = torch.cat(y_l)

        # share features between processes
        feats_all = torch.empty(
            misc.get_world_size(),
            feats.size(0),
            feats.size(1),
            dtype=feats.dtype,
            device=feats.device,
        )

        feats_all = feats_all.contiguous()

        output_l = list(feats_all.unbind(0))
        output_all_reduce = torch.distributed.all_gather(output_l, feats, async_op=True)
        output_all_reduce.wait()

        # update storage feature matrix
        if misc.get_rank() == 0:
            if use_cuda:
                features.index_copy_(0, index_all, torch.cat(output_l))
            else:
                features.index_copy_(0, index_all.cpu(), torch.cat(output_l).cpu())
    return features


@torch.no_grad()
def knn_classifier(train_features, train_labels, test_features, test_labels, k, T, num_classes=1000, num_chunks=500, euclidean=False):
    top1, top5, total = 0.0, 0.0, 0
    train_features = train_features.t()
    num_test_images = test_labels.shape[0]
    imgs_per_chunk = num_test_images // num_chunks
    retrieval_one_hot = torch.zeros(k, num_classes).to(train_features.device)
    for idx in range(0, num_test_images, imgs_per_chunk):
        # get the features for test images
        features = test_features[
            idx : min((idx + imgs_per_chunk), num_test_images), :
        ]
        targets = test_labels[idx : min((idx + imgs_per_chunk), num_test_images)]
        batch_size = targets.shape[0]

        # calculate the dot product and compute top-k neighbors
        similarity = torch.mm(features, train_features)
        if euclidean:
            f_norm = features.pow(2).sum(dim=1, keepdim=True)
            tf_norm = train_features.pow(2).sum(dim=0, keepdim=True)
            similarity = -(f_norm + tf_norm - 2 * similarity)/tf_norm.mean()
        distances, indices = similarity.topk(k, largest=True, sorted=True)
        candidates = train_labels.view(1, -1).expand(batch_size, -1)
        retrieved_neighbors = torch.gather(candidates, 1, indices)

        retrieval_one_hot.resize_(batch_size * k, num_classes).zero_()
        retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1)
        distances_transform = distances.clone().div_(T).exp_()
        probs = torch.sum(
            torch.mul(
                retrieval_one_hot.view(batch_size, -1, num_classes),
                distances_transform.view(batch_size, -1, 1),
            ),
            1,
        )
        _, predictions = probs.sort(1, True)

        # find the predictions that match the target
        correct = predictions.eq(targets.data.view(-1, 1))
        top1 = top1 + correct.narrow(1, 0, 1).sum().item()
        top5 = top5 + correct.narrow(1, 0, min(5, k)).sum().item()  # top5 does not make sense if k < 5
        total += targets.size(0)
    top1 = top1 * 100.0 / total
    top5 = top5 * 100.0 / total
    return top1, top5


def knn_evaluation_pipeline(
    model,
    dataset_train_knn,
    data_loader_train_knn,
    dataset_val_knn,
    data_loader_val_knn,
    avg_pooling,
    temperature,
    nb_knn):
    model.eval()
    if isinstance(temperature, float):
        temperature = [temperature,]
    assert isinstance(temperature, (list, tuple))    
    # ============ extract features ============
    print("Extracting features for train set...")
    train_features = extract_features(model, data_loader_train_knn, avg_pooling=avg_pooling, use_cuda=True)
    print("Extracting features for val set...")
    test_features = extract_features(model, data_loader_val_knn, avg_pooling=avg_pooling, use_cuda=True)

    if misc.get_rank() == 0:
        train_features = nn.functional.normalize(train_features, dim=1, p=2)
        test_features = nn.functional.normalize(test_features, dim=1, p=2)

    train_labels = torch.tensor([s[-1] for s in dataset_train_knn.samples]).long()
    test_labels = torch.tensor([s[-1] for s in dataset_val_knn.samples]).long()
    knn_results = {'k-NN':{}}
    if misc.get_rank() == 0:
        train_features = train_features.cuda()
        test_features = test_features.cuda()
        train_labels = train_labels.cuda()
        test_labels = test_labels.cuda()

        best_top1, best_top5 = 0, 0
        for T in temperature:
            print(f"Features are ready!\nStart the k-NN classification with T={T}.")
            for k in nb_knn:
                top1, top5 = knn_classifier(train_features, train_labels,
                    test_features, test_labels, k, T, euclidean=False)
                print(f"==> {k}-NN classifier result: Top1: {top1}, Top5: {top5}")
                if top1 > best_top1:
                    best_top1 = top1
                if top5 > best_top5:
                    best_top5 = top5
            knn_results['k-NN'].update({k:{'top1':best_top1, 'top5':best_top5}})

    dist.barrier()
    return knn_results
   