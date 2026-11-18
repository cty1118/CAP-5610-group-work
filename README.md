# CAP-5610-group-work

CAP 5610 project: sentence-role classification in medical abstracts on
PubMed 20k RCT ([armanc/pubmed-rct20k](https://huggingface.co/datasets/armanc/pubmed-rct20k)).
We compare MLP (TF-IDF), TextCNN, BiLSTM and fine-tuned BiomedBERT, plus current LLMs.

## Layout

| Path | What | Owner |
|---|---|---|
| `rct/data.py` | Shared pipeline: splits, label ids, position features, vocab, token ids, TF-IDF, GloVe | Both |
| `rct/evaluate.py` | Shared metrics, prediction files, confusion plots, summary table | Both |
| `rct/models.py` | TextCNN and BiLSTM | Tianyu |
| `train_nn.py` | Train / evaluate TextCNN or BiLSTM | Tianyu |
| `sweep_nn.py` | One-at-a-time hyperparameter search for CNN / LSTM | Tianyu |
| MLP, BiomedBERT scripts | to add | Minghao |
| `results/<run>/` | `*_metrics.json`, `*_predictions.csv`, `*_confusion.png` per run | |
| `pubmed_rct_eda.py`, `eda_output/` | EDA | |
| `1026-1.pptx`, `5610.docx` | Proposal slides, notes | |

## Setup

```bash
pip install -r requirements.txt
```

The dataset downloads to `./data/hf_cache` on first use (ignored by git).

## Shared conventions (every model follows these)

- **Labels**: `rct.data.LABELS` order, ids 0–4: Background, Objective, Methods, Results, Conclusions.
- **Position features**: `rct.data.position_features(split)` → 4 columns
  (`sentence_id / total_lines`, is_first, is_last, `total_lines / 30`). The ablation drops all of them.
- **Splits**: tune on `validation` only. Score `test` once, with the final config.
- **Primary metric**: macro-F1; accuracy, weighted-F1, per-class F1 and confusion matrix are saved too.
- **Results**: call `rct.evaluate.save_run(...)`, or write a CSV with columns `idx,pred`
  (`idx` = row index in the Hugging Face split, `pred` = label name or id) and score it:

```bash
python -m rct.evaluate score --pred-file bert_test.csv --split test --name bert_final
python -m rct.evaluate summary --out results/summary.md
```

A prediction file may cover only part of a split, which is how the LLM comparison on a test sample is scored.

## CNN and LSTM (Methods 1 and 2)

```bash
python train_nn.py --model cnn --subset 5000 --epochs 1   # smoke test, under a minute
python train_nn.py --model cnn                            # proposal start config
python train_nn.py --model lstm --pooling attention
python train_nn.py --model cnn --no-position              # position ablation
python sweep_nn.py --model cnn                            # search space from the slides
python sweep_nn.py --model lstm --epochs 6
python train_nn.py --model cnn --name cnn_final <best args> --eval-test   # once, at the end
```

GloVe: `wget https://nlp.stanford.edu/data/glove.6B.zip && unzip glove.6B.zip`, then
`--embedding glove --glove-path glove.6B.300d.txt` (or `sweep_nn.py --glove-path ...`).

Device is picked automatically (CUDA, then Apple MPS, then CPU). On an M-series Mac the CNN takes about
a minute per epoch; the BiLSTM is much slower there, so run LSTM sweeps on Colab.

### Colab

```python
!git clone https://github.com/cty1118/CAP-5610-group-work.git
%cd CAP-5610-group-work
!pip install -q datasets scikit-learn
!python sweep_nn.py --model lstm
!python -m rct.evaluate summary
```

Download `results/` afterwards (or commit it) so runs from both of us end up in one summary table.

## EDA

```bash
python pubmed_rct_eda.py
```
