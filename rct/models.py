"""TextCNN and BiLSTM sentence classifiers (Methods 1 and 2).

Both take (ids, lengths, pos):
    ids      (B, L) int64 token ids, 0 = padding
    lengths  (B,)   int64 true lengths
    pos      (B, P) float32 position features (P = 0 when the ablation drops them)
and return (B, NUM_CLASSES) logits.
"""

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from rct.data import NUM_CLASSES, PAD_ID


def make_embedding(vocab_size, emb_dim, pretrained=None, freeze=False) -> nn.Embedding:
    emb = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_ID)
    if pretrained is not None:
        emb.weight.data.copy_(torch.as_tensor(pretrained))
        emb.weight.requires_grad = not freeze
    return emb


class TextCNN(nn.Module):
    """Kim (2014): parallel Conv1D filters of several widths, max-pooled over time."""

    def __init__(self, vocab_size, emb_dim=128, filter_widths=(3, 4, 5), n_filters=100,
                 dropout=0.5, pos_dim=0, num_classes=NUM_CLASSES, pretrained=None, freeze_emb=False):
        super().__init__()
        self.embedding = make_embedding(vocab_size, emb_dim, pretrained, freeze_emb)
        self.convs = nn.ModuleList(nn.Conv1d(emb_dim, n_filters, w) for w in filter_widths)
        self.min_len = max(filter_widths)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(n_filters * len(filter_widths) + pos_dim, num_classes)

    def forward(self, ids, lengths, pos):
        if ids.size(1) < self.min_len:  # sentences shorter than the widest filter
            ids = nn.functional.pad(ids, (0, self.min_len - ids.size(1)), value=PAD_ID)
        x = self.embedding(ids).transpose(1, 2)  # (B, E, L)
        pooled = [torch.relu(conv(x)).amax(dim=2) for conv in self.convs]
        h = torch.cat(pooled + [pos], dim=1)
        return self.fc(self.dropout(h))


class BiLSTM(nn.Module):
    """Bidirectional LSTM; sentence vector = final states or additive attention pooling."""

    def __init__(self, vocab_size, emb_dim=128, hidden=128, layers=1, pooling="last",
                 dropout=0.3, pos_dim=0, num_classes=NUM_CLASSES, pretrained=None, freeze_emb=False):
        super().__init__()
        assert pooling in ("last", "attention")
        self.pooling = pooling
        self.embedding = make_embedding(vocab_size, emb_dim, pretrained, freeze_emb)
        self.emb_dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(emb_dim, hidden, num_layers=layers, batch_first=True,
                            bidirectional=True, dropout=dropout if layers > 1 else 0.0)
        if pooling == "attention":
            self.att = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1, bias=False))
        self.head = nn.Sequential(
            nn.Linear(2 * hidden + pos_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, ids, lengths, pos):
        x = self.emb_dropout(self.embedding(ids))
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, (h, _) = self.lstm(packed)
        if self.pooling == "last":
            sent = torch.cat([h[-2], h[-1]], dim=1)  # last layer, forward + backward
        else:
            out, _ = pad_packed_sequence(out, batch_first=True)  # (B, T, 2H), T = max length in batch
            scores = self.att(out).squeeze(-1)
            mask = torch.arange(out.size(1), device=out.device)[None, :] < lengths.to(out.device)[:, None]
            weights = torch.softmax(scores.masked_fill(~mask, float("-inf")), dim=1)
            sent = (weights.unsqueeze(-1) * out).sum(dim=1)
        return self.head(torch.cat([sent, pos], dim=1))


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
