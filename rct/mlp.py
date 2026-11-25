"""MLP on TF-IDF features (Method 3).

The input is a sparse TF-IDF vector (1-2-grams) plus the 4 position features.
The first dense layer is written as an nn.EmbeddingBag over the non-zero TF-IDF
entries (weighted by their TF-IDF values) plus a small Linear on the position
features. That is exactly Linear([tfidf, pos]) -> h1, but it never builds the
50k-wide dense input, so training on CPU stays fast.

    x = relu(W_tfidf @ tfidf + W_pos @ pos + b)   # layer 1
    x = relu(W_k @ dropout(x) + b_k)              # layers 2..depth
    logits = W_out @ dropout(x) + b_out
"""

import numpy as np
import torch
from torch import nn

from rct.data import NUM_CLASSES


class TfidfMLP(nn.Module):
    def __init__(self, n_features, hidden=(512, 256), dropout=0.5, pos_dim=0, num_classes=NUM_CLASSES):
        super().__init__()
        assert len(hidden) >= 1
        self.tfidf = nn.EmbeddingBag(n_features, hidden[0], mode="sum")
        self.pos = nn.Linear(pos_dim, hidden[0], bias=False) if pos_dim else None
        self.bias = nn.Parameter(torch.zeros(hidden[0]))
        # same init as nn.Linear(n_features + pos_dim, hidden[0])
        bound = 1.0 / np.sqrt(n_features + pos_dim)
        nn.init.uniform_(self.tfidf.weight, -bound, bound)
        nn.init.uniform_(self.bias, -bound, bound)
        if self.pos is not None:
            nn.init.uniform_(self.pos.weight, -bound, bound)

        layers = []
        for h_in, h_out in zip(hidden[:-1], hidden[1:]):
            layers += [nn.Dropout(dropout), nn.Linear(h_in, h_out), nn.ReLU()]
        layers += [nn.Dropout(dropout), nn.Linear(hidden[-1], num_classes)]
        self.rest = nn.Sequential(*layers)

    def forward(self, indices, offsets, weights, pos):
        h = self.tfidf(indices, offsets, per_sample_weights=weights) + self.bias
        if self.pos is not None:
            h = h + self.pos(pos)
        return self.rest(torch.relu(h))


def csr_batch(X, rows, device):
    """EmbeddingBag inputs (indices, offsets, weights) for the given rows of a CSR matrix."""
    sub = X[rows]
    indices = torch.from_numpy(sub.indices.astype(np.int64)).to(device)
    offsets = torch.from_numpy(sub.indptr[:-1].astype(np.int64)).to(device)
    weights = torch.from_numpy(sub.data.astype(np.float32)).to(device)
    return indices, offsets, weights
