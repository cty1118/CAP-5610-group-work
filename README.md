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
| Method 3: MLP on TF-IDF | Minghao | to add |
| Method 4: BiomedBERT | Minghao | to add |
| LLM comparison (gpt-6.2-sol) | Tianyu | done, on a 200-abstract test sample |
| Error analysis, confidence intervals | Both | done for CNN, BiLSTM, LLM; rerun once MLP and BERT are in |
| Final slides and report | Both | to do |

## Results so far

Test set, scored once with the final config (primary metric macro-F1; weighted-F1 is the
metric to compare with Dernoncourt & Lee 2017):

| Model | Final config | Macro-F1 | Accuracy | Weighted-F1 | Params |
|---|---|---:|---:|---:|---:|
| TextCNN | GloVe 300-d, widths 2,3,4, 100 filters, dropout 0.3 | 80.88 | 86.37 | 86.19 | 13.4M |
| BiLSTM | random 128-d, hidden 128, 1 layer, attention pooling, dropout 0.3 | 84.55 | 89.76 | 89.62 | 5.9M |

Shared 200-abstract test sample (2,328 sentences), 95% bootstrap CI over abstracts:

| Model | Macro-F1 | 95% CI |
|---|---:|---|
| TextCNN | 80.89 | [78.2, 83.5] |
| BiLSTM | 83.12 | [80.6, 85.5] |
| gpt-6.2-sol, sentence + position, 0-shot | 84.10 | [82.1, 86.0] |
| gpt-6.2-sol, whole abstract, 0-shot | 85.10 | [83.1, 87.0] |
| gpt-6.2-sol, whole abstract, 3-shot | 85.84 | [83.8, 87.7] |

Position ablation (validation macro-F1): CNN 81.15 → 77.14, BiLSTM 85.04 → 78.29.
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
| MLP, BiomedBERT scripts | to add | Minghao |
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

## Setup

```bash
pip install -r requirements.txt
```

The dataset downloads to `./data/hf_cache` on first use.

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
