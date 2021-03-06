# coding=utf-8
import torch
from torch.nn import functional as F


def cosine_dist(x, y):
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(2).expand(n, d, m)
    y = y.unsqueeze(2).expand(m, d, n).permute(2, 1, 0)
    return 1 - F.cosine_similarity(x, y)


def euclidean_dist(x, y):
    """
    Compute euclidean distance between two tensors
    """
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)


def prototypical_loss(input, target, n_support, **args):
    """
    Inspired by https://github.com/jakesnell/prototypical-networks/blob/master/protonets/models/few_shot.py

    Compute the barycentres by averaging the features of n_support
    samples for each class in target, computes then the distances from each
    samples' features to each one of the barycentres, computes the
    log_probability for each n_query samples for each one of the current
    classes, of appartaining to a class c, loss and accuracy are then computed and returned
    """

    def supp_idxs(c):
        return target.eq(c).nonzero()[:n_support].squeeze(1)

    classes = torch.unique(target)
    n_classes = len(classes)
    # assuming n_query, n_target constants
    n_query = target.eq(classes[0].item()).sum().item() - n_support

    support_idxs = list(map(supp_idxs, classes))

    prototypes = torch.stack([input[idx_list].mean(0) for idx_list in support_idxs])
    query_idxs = torch.stack(list(map(lambda c: target.eq(c).nonzero()[n_support:], classes))).view(-1)

    query_samples = input[query_idxs]
    if args['distance'] == 'euclidean':
        dists = euclidean_dist(query_samples, prototypes)
    elif args['distance'] == 'cosine':
        dists = cosine_dist(query_samples, prototypes)

    log_p_y = F.log_softmax(-dists, dim=1).view(n_classes, n_query, -1)

    target_inds = torch.arange(0, n_classes).view(n_classes, 1, 1).to(args['device'])
    target_inds = target_inds.expand(n_classes, n_query, 1).long()

    loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
    _, y_hat = log_p_y.max(2)
    acc_val = y_hat.eq(target_inds.squeeze()).float().mean()

    return loss_val,  acc_val
