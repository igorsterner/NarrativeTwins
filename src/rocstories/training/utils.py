import random

import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def infonce(a, b, temperature):

    batch_size = a.size(0)

    feats = torch.cat([a.float(), b.float()], dim=0)

    feats = torch.nn.functional.normalize(feats, dim=1)

    sim = feats @ feats.t()

    mask = torch.eye(2 * batch_size, dtype=torch.bool, device=feats.device)
    sim = sim.masked_fill(mask, float("-inf"))
    sim = sim / float(temperature)

    labels = torch.arange(batch_size, device=feats.device)
    labels = torch.cat([labels + batch_size, labels], dim=0)

    return torch.nn.functional.cross_entropy(sim, labels)
