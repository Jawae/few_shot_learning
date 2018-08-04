# Popular implementations in few-shot learning

Refactored by [@hli2020](https://github.com/hli2020). This repo contains:

- Prototypical Networks for Few-shot Learning, denoted as `nips17_proto`. Forked repo.

- Few-shot learning with graph neural networks, denoted as `iclr18_gnn`. Forked repo.

- Learning to compare: relation network for few-shot learning.
[Forked repo](https://github.com/dragen1860/LearningToCompare-Pytorch).

## Overview

- Supported datasets: Omniglot, Mini-ImageNet

- PyTorch `0.4`

- Multi-gpu if necessary

- To run, see the scripts in `scripts` folder. Results will be logged in `output`


## How to run it
TODO.


## TODO list

- [ ] Support tier-ImageNet

- [ ] Merge dataset processing unified within the repo (for now, there is a `gnn_specific`)

- [ ] Support log visualizations in Visdom and/or TensorboardX