# CAP-5610-group-work

CAP 5610 project: sentence-role classification in medical abstracts on
PubMed 20k RCT ([armanc/pubmed-rct20k](https://huggingface.co/datasets/armanc/pubmed-rct20k)).
We compare MLP (TF-IDF), TextCNN, BiLSTM and fine-tuned BiomedBERT, plus a current LLM.

## Status

| Part | Owner | Status |
|---|---|---|
| Shared pipeline and evaluation (`rct/`) | Both | done |
| Method 1: TextCNN | Tianyu | done: tuned, GloVe comparison, position ablation, test scored |
| Method 2: BiLSTM | Tianyu | done: tuned, position ablation, test scored |
| Method 3: MLP on TF-IDF | Minghao | done: tuned, position ablation, test scored |
| Method 4: BiomedBERT | Minghao | done: LR / epochs / max_len sweep on a 30k subset, DistilBERT and BERT-base comparisons, position ablation, test scored |
| LLM comparison (gpt-6.2-sol) | Tianyu | done, on a 200-abstract test sample |
| Error analysis, confidence intervals | Both | done for all four models and the LLM |
| Final slides and report | Both | to do |

## Results so far

Test set, scored once with the final config (primary metric macro-F1; weighted-F1 is the
metric to compare with Dernoncourt & Lee 2017):

| Model | Final config | Macro-F1 | Accuracy | Weighted-F1 | Params |
|---|---|---:|---:|---:|---:|
| MLP | TF-IDF 1–2-grams 50k + position, 512 → 256, dropout 0.5, Adam 3e-4 | 84.08 | 89.08 | 88.99 | 25.7M |
| TextCNN | GloVe 300-d, widths 2,3,4, 100 filters, dropout 0.3 | 80.88 | 86.37 | 86.19 | 13.4M |
| BiLSTM | random 128-d, hidden 128, 1 layer, attention pooling, dropout 0.3 | 84.55 | 89.76 | 89.62 | 5.9M |
| BiomedBERT | [CLS] + position, lr 2e-5, 3 epochs (best: 2), max_len 128, batch 32 | **84.59** | **89.83** | **89.70** | 109.5M |
| BERT-base (comparison) | same recipe, general-domain `bert-base-uncased`, same size as BiomedBERT | 84.05 | 89.32 | 89.19 | 109.5M |
| DistilBERT (comparison) | same recipe, general-domain `distilbert-base-uncased`, 6 layers | 83.97 | 89.16 | 89.03 | 66.4M |

The top three are within half a point of each other. Background and Objective stay the hard
pair for every model (test F1 about 72–77 and 63–67), because telling them apart mostly needs
the neighbouring sentences, which these sentence-level models do not see.

Shared 200-abstract test sample (2,328 sentences), 95% bootstrap CI over abstracts:

| Model | Macro-F1 | 95% CI |
|---|---:|---|
| TextCNN | 80.89 | [78.2, 83.5] |
| BiLSTM | 83.12 | [80.6, 85.5] |
| MLP | 82.95 | [80.5, 85.4] |
| BiomedBERT | 84.25 | [82.0, 86.6] |
| gpt-6.2-sol, sentence + position, 0-shot | 84.10 | [82.1, 86.0] |
| gpt-6.2-sol, whole abstract, 0-shot | 85.10 | [83.1, 87.0] |
| gpt-6.2-sol, whole abstract, 3-shot | 85.84 | [83.8, 87.7] |

Paired against the BiLSTM on this sample, BiomedBERT is +1.13 [-0.2, +2.6] and the MLP -0.18
[-1.5, +1.1], so neither difference is significant; only the whole-abstract LLM runs beat the BiLSTM
significantly.

PubMed pretraining vs model size (test, full set): BiomedBERT 84.59, same-size general BERT-base
84.05, 6-layer DistilBERT 83.97. Medical pretraining is worth about 0.5 macro-F1 at equal size, and
halving the depth costs only 0.1. On the shared sample BiomedBERT leads BERT-base by 0.98 [-2.2, +0.1],
again not significant ([`results/bootstrap_ci_transformers.md`](results/bootstrap_ci_transformers.md)).

Position ablation (validation macro-F1): CNN 81.15 → 77.14, BiLSTM 85.04 → 78.29,
MLP 84.27 → 77.18, BiomedBERT 85.41 → 83.33. The pretrained Transformer leans on position far
less than the models trained from scratch.
Every run is in [`results/summary.md`](results/summary.md); the error analysis is in
[`results/error_analysis/report.md`](results/error_analysis/report.md).

## Layout

| Path | What | Owner |
|---|---|---|
| `rct/data.py` | Shared pipeline: splits, label ids, position features, vocab, token ids, TF-IDF, GloVe | Both |
| `rct/evaluate.py` | Shared metrics, prediction files, confusion plots, summary table | Both |
| `rct/models.py` | TextCNN and BiLSTM | Tianyu |
| `train_nn.py` | Train / evaluate TextCNN or BiLSTM | Tianyu |
| `sweep_nn.py` | One-at-a-time hyperparameter search for CNN / LSTM | Tianyu |
| `rct/mlp.py` | MLP on sparse TF-IDF + position features | Minghao |
| `train_mlp.py`, `sweep_mlp.py` | Train / sweep the MLP (CPU is enough) | Minghao |
| `train_bert.py`, `sweep_bert.py` | Fine-tune BiomedBERT, BERT-base or DistilBERT; LR sweep on a 30k subset | Minghao |
| `slurm/` | Setup and job scripts for the UCF ARCC Newton GPU cluster | Minghao |
| `llm_compare.py` | LLM comparison through the OpenAI API on a fixed test sample | Tianyu |
| `bootstrap_ci.py` | 95% confidence intervals and paired differences on the shared sample | Tianyu |
| `error_analysis.py` | Error types, Background vs Objective, errors by position, model agreement | Tianyu |
| `results/<run>/` | `*_metrics.json`, `*_predictions.csv`, `*_confusion.png` per run | |
| `results/summary.md` | One table with every run | |
| `results/llm_sample_test_abstracts.txt` | The 200 test abstract ids every LLM comparison uses | |
| `results/bootstrap_ci.md`, `results/error_analysis/` | Confidence intervals; error-analysis report and figure | |
| `pubmed_rct_eda.py`, `eda_output/` | EDA | |
| `1026-1.pptx`, `5610.docx` | Proposal slides, notes | |

Not in git: `data/` (dataset cache), `glove/` (word vectors), `checkpoints/` (model weights),
`.env` (API key).

## Dependencies and installation

### Requirements

- **Python 3.9 or newer.** Tested with Python 3.9.6 on macOS 15 (Apple M4) and with the
  default Python on Google Colab.
- **Hardware**: everything runs on CPU, but a GPU is strongly recommended for training.
  `train_nn.py` picks CUDA, then Apple MPS, then CPU automatically.
  Tested on an Apple M4 (MPS) for the CNN and an NVIDIA T4 on Colab for the BiLSTM.
- **Disk**: about 50 MB for the dataset cache, plus about 1 GB if you use GloVe.

### Python packages

All are in [`requirements.txt`](requirements.txt). Versions we ran with:

| Package | Used for | Tested version |
|---|---|---|
| `torch` | All neural models and training | 2.8.0; 2.15 (CPU, Windows) and 2.14.1+cu126 (Newton) for MLP / BERT |
| `transformers` | BiomedBERT, BERT-base and DistilBERT (`train_bert.py`) | 5.24.1 |
| `datasets` | Downloading PubMed 20k RCT from Hugging Face | 4.5.0, 5.2.0 |
| `numpy` | Arrays, sampling, bootstrap | 2.0.2 |
| `scikit-learn` | Metrics (macro-F1, confusion matrix), TF-IDF | 1.6.1, 1.9.1 |
| `scipy` | Sparse TF-IDF matrices for the MLP (installed with scikit-learn) | 1.18.1 |
| `matplotlib` | Confusion-matrix and error-analysis figures | 3.9.4 |
| `openai` | LLM comparison only (`llm_compare.py`) | 2.48.0 |

Everything else comes from the Python standard library. The MLP and BERT code was run with
Python 3.13 on Windows 11 (MLP, CPU) and Python 3.11 on UCF ARCC Newton (BERT, V100 GPUs).

### Install

```bash
git clone https://github.com/cty1118/CAP-5610-group-work.git
cd CAP-5610-group-work
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

For an NVIDIA GPU, install the matching CUDA build of PyTorch first, following
[pytorch.org/get-started](https://pytorch.org/get-started/locally/). On Colab, `torch` is
already installed, so `pip install -q datasets scikit-learn` is enough.

Quick check that the install works (under a minute, no GPU needed):

```bash
python train_nn.py --model cnn --subset 5000 --epochs 1
```

### External resources

| Resource | Needed for | How to get it |
|---|---|---|
| PubMed 20k RCT dataset | Everything | Downloaded automatically on first run to `./data/hf_cache` |
| GloVe 300-d word vectors | Only the GloVe CNN (`--embedding glove`) | See the GloVe commands in the CNN section below; saved to `./glove/` |
| BiomedBERT / BERT-base / DistilBERT weights | Only `train_bert.py` | Downloaded automatically from Hugging Face on first run (about 440, 440 and 270 MB) |
| OpenAI API key | Only `llm_compare.py` | Create one at platform.openai.com and put it in `.env` (see LLM comparison below) |

None of these are in the repository: the dataset and GloVe are public and re-downloaded,
and the API key must never be committed. `bootstrap_ci.py` and `error_analysis.py` need no
key, because they only read the saved prediction files in `results/`.

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
python sweep_nn.py --model cnn                            # search space from the slides
python sweep_nn.py --model lstm --epochs 6
python train_nn.py --model cnn --no-position              # position ablation
```

Final runs (test scored once):

```bash
python train_nn.py --model cnn --name cnn_final --dropout 0.3 --filter-widths 2,3,4 \
    --embedding glove --glove-path glove/glove.6B.300d.txt --eval-test --save-model
python train_nn.py --model lstm --name lstm_final --pooling attention --epochs 6 --eval-test
```

GloVe 300-d (about 860 MB zipped; nlp.stanford.edu, or the official mirror on Hugging Face):

```bash
mkdir -p glove && curl -L -o glove/glove.6B.zip https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.zip
unzip glove/glove.6B.zip glove.6B.300d.txt -d glove && rm glove/glove.6B.zip
```

Device is picked automatically (CUDA, then Apple MPS, then CPU). On an M-series Mac the CNN takes about
a minute per epoch; the BiLSTM is much slower there, so the LSTM runs were done on Colab (T4).

### Colab

Save results to Google Drive so a disconnect loses nothing:

```python
from google.colab import drive
drive.mount('/content/drive')
!git clone https://github.com/cty1118/CAP-5610-group-work.git
%cd CAP-5610-group-work
!pip install -q datasets scikit-learn
!python sweep_nn.py --model lstm --epochs 6 --results-dir /content/drive/MyDrive/rct_results
!python -m rct.evaluate summary --results-dir /content/drive/MyDrive/rct_results
```

Copy the run folders back into `results/` and commit them, so every run ends up in one summary table.

## MLP and BiomedBERT (Methods 3 and 4)

### MLP on TF-IDF (CPU)

TF-IDF over 1–2-grams (fit on train only, `rct.data.tfidf_features`) plus the 4 position features,
then Dense → ReLU → Dropout layers and a 5-way softmax. The first layer is an `nn.EmbeddingBag`
over the non-zero TF-IDF entries, which is the same linear map as a dense layer on the 50k-wide
vector but much faster on CPU (about a minute per epoch on a 20-core desktop). Batch 256, Adam,
early stopping on validation macro-F1 with patience 2.

```bash
python train_mlp.py --subset 5000 --epochs 1        # smoke test, under a minute
python sweep_mlp.py                                 # slide 10 search space, one setting at a time
python train_mlp.py --name mlp_final --lr 3e-4 --eval-test --save-model
python train_mlp.py --name mlp_final_nopos --lr 3e-4 --no-position   # position ablation
```

Every sweep setting landed within 83.8–84.3 validation macro-F1; combining the small winners
(`results/mlp_lr3e-4_*`) did not beat lowering the learning rate alone, so the final MLP is the
start config (512 → 256, dropout 0.5, 50k features) with Adam 3e-4. It overfits after 1–2 epochs.

### BiomedBERT, BERT-base and DistilBERT (GPU)

`microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext` (Gu et al. 2021); the sentence is
WordPiece-tokenized, the final-layer [CLS] vector is concatenated with the 4 position features and
fed to one linear layer. AdamW, batch 32, 10% linear warmup, weight decay 0.01, gradient clipping 1.0,
mixed precision (fp16 on V100, bf16 on H100). The new linear head gets learning rate 1e-3
(`--head-lr`), because at 2e-5 it would barely learn the position weights in 3 epochs. Validation is
scored after every epoch and the best epoch is kept. `--model bertbase` and `--model distilbert` run
the same recipe on general-domain `bert-base-uncased` (same size) and `distilbert-base-uncased`.

```bash
python train_bert.py --subset 2000 --eval-subset 1000 --epochs 1      # smoke test (CPU works, slowly)
python sweep_bert.py --only lr2e-5,lr3e-5,lr5e-5,e2,len128           # 30k-sentence subset, full validation
python train_bert.py --name bert_final --max-len 128 --eval-test --save-model
python train_bert.py --name bert_bertbase --model bertbase --max-len 128 --eval-test
python train_bert.py --name bert_distilbert --model distilbert --max-len 128 --eval-test
python train_bert.py --name bert_final_nopos --max-len 128 --no-position
```

On the 30k subset all five sweep settings scored 82.2–82.7 validation macro-F1; lr 2e-5 (the start
value) was best among the learning rates and max_len 128 added 0.2, so the final config is lr 2e-5,
3 epochs, max_len 128. Validation peaked after epoch 2 for all three Transformers, so that
epoch's weights were kept. On a V100 one full-data epoch takes about 5.5 minutes for BiomedBERT and 3
for DistilBERT; `bert_final`'s recorded 36 training minutes include a first epoch that ran several
times slower on a busy node.

### Running on UCF ARCC Newton

Password-only SSH, Slurm account `qlou`. From the repo on the Newton login node:

```bash
bash slurm/setup_newton.sh          # once: venv with CUDA PyTorch + Transformers
mkdir -p logs
sbatch --job-name=bert_final --time=01:15:00 slurm/newton_gpu.sbatch \
    python train_bert.py --name bert_final --max-len 128 --eval-test --save-model
```

The login node caps memory at 4 GB, so it cannot import CUDA PyTorch; compute nodes can, and they
have internet, so the first job downloads the dataset and checkpoints. If the shared `qlou` CPU-hour
budget is used up (jobs pending with `AssocGrpCPUMinutesLimit`), add
`--partition=preemptable --qos=preemptable`: that queue is not charged, but jobs can be preempted.
Copy `results/<run>/` back into this repo and commit it.

## LLM comparison

The key goes in `.env` (git-ignored), never in code or chat. Copy the key, then:

```bash
printf 'OPENAI_API_KEY=%s\n' "$(pbpaste)" > .env && chmod 600 .env
```

```bash
python llm_compare.py --model gpt-6.2-sol --setting abstract --max-requests 5   # trial + token estimate
python llm_compare.py --model gpt-6.2-sol --setting abstract                    # whole abstract, 0-shot
python llm_compare.py --model gpt-6.2-sol --setting sentence                    # one sentence + position
python llm_compare.py --model gpt-6.2-sol --setting abstract --shots 3          # 3 training abstracts as examples
```

All runs use the same 200 test abstracts (`results/llm_sample_test_abstracts.txt`, seed 42).
Responses are cached in `results/<run>/responses.jsonl`, so a rerun resumes without paying twice.
The three runs above used about 714k input and 117k output tokens.

To compare another model on exactly the same sentences, re-score its saved test predictions on the sample:

```bash
python llm_compare.py --baselines mlp_final,bert_final    # writes results/mlp_final_on_llm_sample/ etc.
```

## Confidence intervals and error analysis

Both read saved `test_predictions.csv` files only: no training, no API calls.

```bash
python bootstrap_ci.py       # -> results/bootstrap_ci.md / .json
python error_analysis.py     # -> results/error_analysis/report.md and error_by_position.png
```

Adding MLP and BERT once their `*_final` runs exist (after the `--baselines` step above):

```bash
python bootstrap_ci.py --runs "CNN=cnn_final_on_llm_sample" "BiLSTM=lstm_final_on_llm_sample" \
    "MLP=mlp_final_on_llm_sample" "BiomedBERT=bert_final_on_llm_sample" \
    "gpt-6.2-sol abstract 3-shot=llm_gpt-6.2-sol_abstract_3shot"
python error_analysis.py --full "CNN=cnn_final" "BiLSTM=lstm_final" "MLP=mlp_final" "BiomedBERT=bert_final"
```

## EDA

```bash
python pubmed_rct_eda.py
```
