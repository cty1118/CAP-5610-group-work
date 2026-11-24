# Bootstrap confidence intervals (200 test abstracts, 2,328 sentences, 2,000 resamples of whole abstracts)

| Run | Macro-F1 | 95% CI |
|---|---:|---|
| CNN | 80.89 | [78.2, 83.5] |
| BiLSTM | 83.12 | [80.6, 85.5] |
| gpt-6.2-sol sentence 0-shot | 84.10 | [82.1, 86.0] |
| gpt-6.2-sol abstract 0-shot | 85.10 | [83.1, 87.0] |
| gpt-6.2-sol abstract 3-shot | 85.84 | [83.8, 87.7] |

## Difference in macro-F1 vs BiLSTM (paired: same resampled abstracts for both runs)

| Run | Difference | 95% CI | Significant |
|---|---:|---|---|
| CNN | -2.23 | [-3.9, -0.6] | yes |
| gpt-6.2-sol sentence 0-shot | +0.97 | [-0.6, +2.6] | no |
| gpt-6.2-sol abstract 0-shot | +1.97 | [+0.3, +3.6] | yes |
| gpt-6.2-sol abstract 3-shot | +2.71 | [+1.0, +4.4] | yes |

Significant = the 95% interval of the difference does not contain 0.
