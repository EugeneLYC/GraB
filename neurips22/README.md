# GraB

## Introduction
---
This repository contains source code for paper:

[GraB: Finding Provably Better Data Permutations than Random Reshuffling
](https://arxiv.org/abs/2205.10733).

Authors: [Yucheng Lu](https://www.cs.cornell.edu/~yucheng/), Wentao Guo, and [Christopher De Sa](https://www.cs.cornell.edu/~cdesa/)

In the Thirty-sixth Conference on Neural Information Processing Systems ([NeurIPS](https://nips.cc/)) 2022.

This paper proposes a way of ordering examples (data points), named GraB, in epoch-based model training. The main idea is to reorder the examples at the end of every epoch based on the discrepancy (termed in the paper) that minimizes the consecutive gradient errors. In the paper, we've shown GraB outperforms both shuffle once and random reshuffling (widely used in the community) with very little overhead.
Please contact [Yucheng Lu](https://www.cs.cornell.edu/~yucheng/) if you have any questions or suggestions on the paper / code: yl2967@cornell.edu.

## Reproduce the Results
---
Note that this repo contains the experimental code for GraB paper, which is different from the OrderedSampler. To reproduce the results, first
```
export PYTHONPATH=<path to GraB>/neurips22/src/
```
and then,
```
pip install -r requirements.txt
```

## Baselines
---
The implementation supports the following example ordering methods:

* `random_reshuffling`: randomly reshuffle the training data at the beginning of each epoch (used in most applications).
* `shuffle_once`: only randomly shuffle the dataset at the beginning of the first epoch.
* `stale_grad_greedy_sort`: sort the examples using the gradients computed from the previous epoch. ([reference](https://openreview.net/pdf?id=7gWSJrP3opB))
* `fresh_grad_greedy_sort`: sort the examples using the fresh gradients computed at the beginning of the epoch. ([reference](https://openreview.net/pdf?id=7gWSJrP3opB))
* `flipflop`: implementation of a random reshuffling variant. ([reference](https://arxiv.org/pdf/2102.09718.pdf))
* `dm`: the online gradient balancing algorithm, namely GraB.

Scripts to reproduce the results can be found in the `examples` repo.

Detailed comparison among different algorithms can be found in the [GraB](https://arxiv.org/abs/2205.10733) paper.

For more practical usage, please refer to the `OrderedSampler`.
