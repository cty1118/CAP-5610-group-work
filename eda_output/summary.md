# PubMed 20k RCT (armanc/pubmed-rct20k): EDA summary

| Split | Abstracts | Sentences | Imbalance (max/min) |
|---|---:|---:|---:|
| train | 15,000 | 176,642 | 4.28 |
| validation | 2,500 | 29,672 | 4.19 |
| test | 2,500 | 29,578 | 4.24 |

## Class distribution

| Class | train | validation | test |
|---|---:|---:|---:|
| Background | 18,402 (10.42%) | 2,925 (9.86%) | 3,077 (10.4%) |
| Objective | 13,838 (7.83%) | 2,376 (8.01%) | 2,333 (7.89%) |
| Methods | 59,281 (33.56%) | 9,948 (33.53%) | 9,884 (33.42%) |
| Results | 57,953 (32.81%) | 9,841 (33.17%) | 9,713 (32.84%) |
| Conclusions | 27,168 (15.38%) | 4,582 (15.44%) | 4,571 (15.45%) |

## Tokens per sentence

| Split | mean | median | p90 | p95 | p99 | max |
|---|---:|---:|---:|---:|---:|---:|
| train | 26.7 | 24 | 45 | 55 | 80 | 296 |
| validation | 26.77 | 24 | 46 | 55 | 81 | 258 |
| test | 26.54 | 23 | 45 | 55 | 80 | 211 |

## Sentences per abstract

| Split | mean | median | p95 | max |
|---|---:|---:|---:|---:|
| train | 11.78 | 11 | 17 | 31 |
| validation | 11.87 | 12 | 17 | 27 |
| test | 11.83 | 12 | 17 | 28 |

## Vocabulary (train, lowercased whitespace tokens)

- Distinct tokens: 69,126
- Tokens seen at least twice: 43,788
- Validation tokens not in train vocabulary (OOV rate): 1.54%

## Coverage of max_len = 55

- Train sentences with at most 55 tokens: 95.23%
