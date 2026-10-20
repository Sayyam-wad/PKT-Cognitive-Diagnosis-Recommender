import json

import torch
import torch.nn.init
from torch.autograd import Variable


def variable(tensor, gpu):
    if gpu >= 0:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but is not available; use --gpu -1 for CPU")
        return Variable(tensor).to(torch.device("cuda", gpu))
    return Variable(tensor).cpu()


def to_scalar(var):
    return var.detach().view(-1)[0].item()


def save_checkpoint(state, track_list, filename):
    with open(filename + '.json', 'w') as f:
        json.dump(track_list, f)
    torch.save(state, filename + '.model')


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
