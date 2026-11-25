# Bootstrap confidence intervals (200 test abstracts, 2,328 sentences, 2,000 resamples of whole abstracts)

| Run | Macro-F1 | 95% CI |
|---|---:|---|
| BiomedBERT | 84.25 | [82.0, 86.6] |
| BERT-base | 83.27 | [80.9, 85.6] |
| DistilBERT | 83.12 | [80.7, 85.4] |
| BiLSTM | 83.12 | [80.6, 85.5] |
| MLP | 82.95 | [80.5, 85.4] |

## Difference in macro-F1 vs BiomedBERT (paired: same resampled abstracts for both runs)

| Run | Difference | 95% CI | Significant |
|---|---:|---|---|
| BERT-base | -0.98 | [-2.2, +0.1] | no |
| DistilBERT | -1.14 | [-2.5, +0.1] | no |
| BiLSTM | -1.13 | [-2.6, +0.2] | no |
| MLP | -1.31 | [-2.9, +0.2] | no |

Significant = the 95% interval of the difference does not contain 0.
