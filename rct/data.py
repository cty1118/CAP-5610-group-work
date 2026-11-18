"""Shared data pipeline for all four models (MLP, CNN, LSTM, Transformer).

Every model reads the same splits, the same label ids and the same position
features from here, so results are directly comparable.

    from rct.data import load_splits, position_features, build_vocab, encode

    splits = load_splits()                      # {"train": Split, "validation": ..., "test": ...}
    pos = position_features(splits["train"])    # (N, POS_DIM) float32
    vocab = build_vocab(splits["train"].texts)  # word -> id, 0 = <pad>, 1 = <unk>
    ids, lengths = encode(splits["train"].texts, vocab, max_len=55)

Text in the dataset is already tokenized (punctuation separated by spaces) and
numbers are masked as "@", so tokenization is lowercase + whitespace split.
"""

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HF_DATASET = "armanc/pubmed-rct20k"
SPLITS = ["train", "validation", "test"]
LABELS = ["BACKGROUND", "OBJECTIVE", "METHODS", "RESULTS", "CONCLUSIONS"]
LABEL2ID = {lab: i for i, lab in enumerate(LABELS)}
NUM_CLASSES = len(LABELS)

PAD, UNK = "<pad>", "<unk>"
PAD_ID, UNK_ID = 0, 1

# 95th percentile of train sentence length (see eda_output/summary.md): covers 95.2%.
DEFAULT_MAX_LEN = 55

# Position features, in this order:
#   rel_pos     sentence_id / total_lines          (the feature named in the proposal)
#   is_first    1 if the first sentence, else 0
#   is_last     1 if the last sentence, else 0
#   total_norm  total_lines / 30                     (abstract length; max in train is 31)
POS_FEATURE_NAMES = ["rel_pos", "is_first", "is_last", "total_norm"]
POS_DIM = len(POS_FEATURE_NAMES)

DEFAULT_CACHE = Path(__file__).resolve().parent.parent / "data" / "hf_cache"


@dataclass
class Split:
    """One split as parallel lists; index i is one sentence."""

    texts: list
    labels: np.ndarray        # int64 label ids, see LABELS
    abstract_ids: list
    sentence_ids: np.ndarray  # 0-based position inside the abstract
    total_lines: np.ndarray   # number of sentences in the abstract
    index: np.ndarray = None  # row index in the full split (kept through subset())

    def __post_init__(self):
        if self.index is None:
            self.index = np.arange(len(self.texts))

    def __len__(self):
        return len(self.texts)

    def subset(self, n: int, seed: int = 0) -> "Split":
        """Random sample of n sentences (for quick smoke tests)."""
        if n >= len(self):
            return self
        idx = np.sort(np.random.default_rng(seed).choice(len(self), n, replace=False))
        return Split(
            texts=[self.texts[i] for i in idx],
            labels=self.labels[idx],
            abstract_ids=[self.abstract_ids[i] for i in idx],
            sentence_ids=self.sentence_ids[idx],
            total_lines=self.total_lines[idx],
            index=self.index[idx],
        )


def load_splits(cache_dir=DEFAULT_CACHE) -> dict:
    """Load train / validation / test from Hugging Face as {name: Split}.

    The row order of each split is kept as-is, so index i in a split always
    refers to the same sentence. Prediction files (rct.evaluate) rely on this.
    """
    from datasets import load_dataset

    ds = load_dataset(HF_DATASET, cache_dir=str(cache_dir))
    out = {}
    for name in SPLITS:
        rows = ds[name]
        abstract_ids = list(rows["abstract_id"])
        total = Counter(abstract_ids)
        out[name] = Split(
            texts=list(rows["text"]),
            labels=np.array([LABEL2ID[lab.upper()] for lab in rows["label"]], dtype=np.int64),
            abstract_ids=abstract_ids,
            sentence_ids=np.array(rows["sentence_id"], dtype=np.int64),
            total_lines=np.array([total[a] for a in abstract_ids], dtype=np.int64),
        )
    return out


def position_features(split: Split) -> np.ndarray:
    """(N, POS_DIM) float32 matrix; columns follow POS_FEATURE_NAMES."""
    sid = split.sentence_ids.astype(np.float32)
    tot = split.total_lines.astype(np.float32)
    return np.stack(
        [
            sid / tot,
            (sid == 0).astype(np.float32),
            (sid == tot - 1).astype(np.float32),
            tot / 30.0,
        ],
        axis=1,
    ).astype(np.float32)


def tokenize(text: str) -> list:
    return text.lower().split()


def build_vocab(train_texts, min_freq: int = 2, max_size: int = None) -> dict:
    """word -> id from the training texts only. Ids 0 and 1 are <pad> and <unk>."""
    counts = Counter(tok for t in train_texts for tok in tokenize(t))
    words = [w for w, c in counts.most_common(max_size) if c >= min_freq]
    vocab = {PAD: PAD_ID, UNK: UNK_ID}
    for w in words:
        vocab[w] = len(vocab)
    return vocab


def encode(texts, vocab: dict, max_len: int = DEFAULT_MAX_LEN):
    """Token ids padded / truncated to max_len, plus true lengths (at least 1)."""
    ids = np.full((len(texts), max_len), PAD_ID, dtype=np.int64)
    lengths = np.empty(len(texts), dtype=np.int64)
    for i, t in enumerate(texts):
        toks = [vocab.get(w, UNK_ID) for w in tokenize(t)][:max_len] or [UNK_ID]
        ids[i, : len(toks)] = toks
        lengths[i] = len(toks)
    return ids, lengths


def tfidf_features(splits: dict, max_features: int = 50_000, ngram_range=(1, 2)):
    """TF-IDF vectors for the MLP: fit on train, transform all splits.

    Returns ({split: sparse matrix}, fitted vectorizer). Append position_features
    (e.g. with scipy.sparse.hstack) to match the other models.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer(
        lowercase=True,
        token_pattern=r"\S+",
        ngram_range=ngram_range,
        max_features=max_features,
        sublinear_tf=True,
        dtype=np.float32,
    )
    out = {"train": vec.fit_transform(splits["train"].texts)}
    for name in splits:
        if name != "train":
            out[name] = vec.transform(splits[name].texts)
    return out, vec


def load_glove(path, vocab: dict, dim: int = 300, seed: int = 0) -> np.ndarray:
    """Embedding matrix (len(vocab), dim) from a GloVe .txt file.

    Words missing from GloVe get small random vectors; <pad> stays zero.
    Download (Colab): wget https://nlp.stanford.edu/data/glove.6B.zip && unzip glove.6B.zip
    """
    rng = np.random.default_rng(seed)
    emb = rng.normal(0, 0.1, (len(vocab), dim)).astype(np.float32)
    emb[PAD_ID] = 0
    found = 0
    with open(path, encoding="utf8") as f:
        for line in f:
            word, *vals = line.rstrip().split(" ")
            idx = vocab.get(word)
            if idx is not None and len(vals) == dim:
                emb[idx] = np.asarray(vals, dtype=np.float32)
                found += 1
    print(f"GloVe: {found:,} / {len(vocab):,} vocabulary words found")
    return emb
