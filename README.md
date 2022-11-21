# OrderedSampler

This repository provides a useful [pytorch-based](https://pytorch.org/) batch sampler for [map-styple datasets](https://pytorch.org/docs/stable/data.html#map-style-datasets), named OrderedSampler. When model is trained with many epochs,  OrderedSampler is able to find better example orderings (than random reshuffling) in each epoch, and let the training converge faster. 
The ordering is based on a technique named gradient balancing.
The technical details can be found in this [NeurIPS'22 paper](https://arxiv.org/abs/2205.10733) (authored by [Yucheng Lu](https://www.cs.cornell.edu/~yucheng/), Wentao Guo, and [Christopher De Sa](https://www.cs.cornell.edu/~cdesa/)). Please contact [Yucheng Lu](https://www.cs.cornell.edu/~yucheng/) if you have any questions or suggestions on the paper / code: yl2967@cornell.edu.

For reproducing the results in [NeurIPS'22 paper](https://arxiv.org/abs/2205.10733), please refer to the ``neurips22`` repository.

**Watch Out!**
Note that OrderedSampler orders finite number of examples after each full scan (epoch), and so currently it does not support [iterable-style datasets](https://pytorch.org/docs/stable/data.html#iterable-style-datasets) since there is no notion of ordering finite number of examples.

## Installation
---
There are two ways of using `OrderedSampler`, the first is to install the package directly:
```
python setup.py install
```
while the second is to include the path to OrderedSampler
```
export PYTHONPATH=<path to orderedsampler>/src/
```
If you use the second method, then the requirements need to be installed manually.
```
pip install -r requirements.txt
```

## Usage

### Arguments
* `data_source` (Dataset): Dataset to sample from.
* `batch_size` (int): Size of mini-batch (default: 1).
* `order_level` (int): Granularity of ordering (default: 1).
* `drop_last` (bool): If ``True``, the sampler will drop the last batch if its size would be less than ``batch_size`` (default: False).
* `init_order_random` (bool): If ``True``, the initial order (first scan of the dataset) will be random (default: True).
* `model` (nn.Module): Model to train (default: None).
* `lossfunc`: (nn.Module): Loss function used during the training (default: None).
* `debug` (bool): Whether to turn on the debugging mode (default: False).
* `balance_type` (str): the balancing algorithm to use. Currently ``pair_balance`` and ``mean_balance`` are supported. Note that if ``mean_balance`` is used, the stale gradient mean from previous epoch will be applied. If the training involves large learning rate or contains few epochs, ``pair_balance`` is recommended (default: pair_balance).
* `prob_balance` (bool): If ``True``, probabilistic balancing will be performed. This is useful when the data is highly adversrial. for technical details, please refer to [this paper](https://arxiv.org/abs/2006.14009) (default: False).

### Example
---
The OrderedSampler is easy to be integrated into any training script. For instance,

```python
from orderedsampler import OrderedSampler

model = torch.nn.Linear(784, 10)
lossfunc = torch.nn.CrossEntropyLoss()
dataset = torchvision.datasets.MNIST('./data', train=True, ...)

#################################################################
sampler = OrderedSampler(dataset, ...)                   # <- add
model, lossfunc = sampler.model, sampler.lossfunc        # <- add
#################################################################

dataloader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler)

for epoch in range(10):
    for _, (x, y) in enumerate(dataloader):
        optimizer.zero_grad()
        loss = lossfunc(model(x), y)
        loss.backward()
        ######################
        sampler.step()# <- add
        ######################
        optimizer.step()
```
A full example script of training logistic regression on MNIST dataset can be found in ``./examples/train_logistic_regression.py``.

## Citation
If you find this code useful, please consider citing us:
```
@inproceedings{
    lu2022grab,
    title={GraB: Finding Provably Better Data Permutations than Random Reshuffling},
    author={Yucheng Lu and Wentao Guo and Christopher De Sa},
    booktitle={Advances in Neural Information Processing Systems},
    editor={Alice H. Oh and Alekh Agarwal and Danielle Belgrave and Kyunghyun Cho},
    year={2022},
    url={https://openreview.net/forum?id=nDemfqKHTpK}
}
```

```
@inproceedings{
    lu2022a,
    title={A General Analysis of Example-Selection for Stochastic Gradient Descent},
    author={Yucheng Lu and Si Yi Meng and Christopher De Sa},
    booktitle={International Conference on Learning Representations},
    year={2022},
    url={https://openreview.net/forum?id=7gWSJrP3opB}
}
```
