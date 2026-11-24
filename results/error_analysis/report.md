# Error analysis

Full test set (29,578 sentences): CNN, BiLSTM. Shared 200-abstract sample (2,328 sentences): CNN, BiLSTM, GPT sentence 0-shot, GPT abstract 3-shot.

## 1. What kind of errors

### Full test set

**CNN**: 4,032 errors in 29,578 sentences (13.6%); Background ↔ Objective = 1,286 (32% of errors)

| True → predicted | Errors | Share |
|---|---:|---:|
| Objective → Background | 875 | 22% |
| Results → Methods | 800 | 20% |
| Background → Objective | 411 | 10% |
| Conclusions → Results | 381 | 9% |
| Results → Conclusions | 339 | 8% |
| Methods → Results | 316 | 8% |

**BiLSTM**: 3,028 errors in 29,578 sentences (10.2%); Background ↔ Objective = 1,313 (43% of errors)

| True → predicted | Errors | Share |
|---|---:|---:|
| Objective → Background | 941 | 31% |
| Results → Methods | 475 | 16% |
| Background → Objective | 372 | 12% |
| Methods → Results | 348 | 11% |
| Conclusions → Results | 289 | 10% |
| Results → Conclusions | 210 | 7% |

### LLM sample

**GPT sentence 0-shot**: 232 errors in 2,328 sentences (10.0%); Background ↔ Objective = 123 (53% of errors)

| True → predicted | Errors | Share |
|---|---:|---:|
| Objective → Background | 63 | 27% |
| Background → Objective | 60 | 26% |
| Results → Methods | 45 | 19% |
| Methods → Results | 20 | 9% |
| Background → Methods | 10 | 4% |
| Conclusions → Results | 8 | 3% |

**GPT abstract 3-shot**: 186 errors in 2,328 sentences (8.0%); Background ↔ Objective = 124 (67% of errors)

| True → predicted | Errors | Share |
|---|---:|---:|
| Objective → Background | 63 | 34% |
| Background → Objective | 61 | 33% |
| Results → Methods | 39 | 21% |
| Methods → Objective | 6 | 3% |
| Methods → Results | 4 | 2% |
| Methods → Conclusions | 3 | 2% |

## 2. Background vs Objective

In the 2,500 test abstracts, 51% contain a Background sentence, 63% an Objective sentence and only 14% both. The first sentence is Background in 50% of abstracts and Objective in 50%. Which label the opening sentences get depends largely on whether the authors used a 'Background' or an 'Objective' heading.

| Model | Objective → Background | …in abstracts with no Background | Background → Objective | …in abstracts with no Objective |
|---|---:|---:|---:|---:|
| CNN | 875 | 89% | 411 | 90% |
| BiLSTM | 941 | 90% | 372 | 96% |
| GPT sentence 0-shot | 63 | 100% | 60 | 97% |
| GPT abstract 3-shot | 63 | 100% | 61 | 100% |

## 3. Where in the abstract

![Error rate by position](error_by_position.png)

| Position | CNN error % | BiLSTM error % | Sentences |
|---|---:|---:|---:|
| 1 | 25.4 | 22.5 | 2,500 |
| 2 | 25.0 | 22.0 | 2,500 |
| 3 | 17.9 | 15.1 | 2,500 |
| 4 | 12.4 | 10.0 | 2,500 |
| 5 | 10.6 | 8.3 | 2,497 |
| 6 | 10.7 | 7.9 | 2,486 |
| 7 | 9.5 | 6.2 | 2,457 |
| 8 | 8.9 | 5.8 | 2,388 |
| 9 | 10.0 | 5.8 | 2,224 |
| 10 | 10.0 | 5.2 | 1,965 |
| 11+ | 11.1 | 6.5 | 5,561 |

## 4. Do the models fail on the same sentences? (LLM sample)

| Models wrong | Sentences | Share |
|---|---:|---:|
| 0 | 1912 | 82.1% |
| 1 | 151 | 6.5% |
| 2 | 85 | 3.7% |
| 3 | 49 | 2.1% |
| 4 | 131 | 5.6% |

Sentences every model gets wrong, by true label: Objective 60, Results 36, Background 27, Methods 6, Conclusions 2

Of these 131, 119 get the same wrong label from every model, a hint that the gold label rather than the models may be the odd one out.

Random examples (predictions in the order CNN, BiLSTM, GPT sentence 0-shot, GPT abstract 3-shot):

| Pos | Gold | Predicted | Sentence |
|---|---|---|---|
| 3/14 | Results | Methods / Methods / Methods / Methods | @,@ infants aged @-@ wk and @,@ children aged @-@ mo were randomized to receive three doses of RTS , S/AS@ or comparator vaccine . |
| 1/13 | Objective | Background / Background / Background / Background | Child pedestrian injury is a global public health challenge . |
| 5/11 | Results | Methods / Methods / Methods / Methods | Circulating EPC ( cells positive for CD@ , CD@ and kinase insert domain receptor - KDR cell-surface antigens by flow cytometry ) number was evaluated in @ healthy subjects ( @ F/@ M , @ @ years ) , randomized to recei... |
| 3/10 | Background | Methods / Methods / Methods / Methods | This prospective randomized placebo-controlled multicenter trial is designed to evaluate the efficacy of ferric carboxymaltose administration with the primary end point of successful hemoglobin level increase by @ g/d... |
| 2/19 | Objective | Background / Background / Background / Background | Adequate rehydration following physical activity may not occur , consequently resulting in the consumption of alcohol in a dehydrated state . |
| 3/10 | Objective | Background / Background / Background / Background | These biochemical changes have not been correlated with changes in whole tendon structure on imaging . |
| 2/11 | Objective | Background / Background / Background / Background | Although several treatments provide symptomatic relief , there is no effective treatment to prevent or treat radiation cystitis . |
| 6/13 | Results | Methods / Methods / Methods / Methods | Quality-adjusted lifeyears were calculated by utilizing community-based preferences . |
| 1/9 | Objective | Background / Background / Background / Background | Certain plant polysaccharides may provide psychological health benefits . |
| 3/12 | Objective | Background / Background / Background / Background | Acetazolamide effectively counteracts nocturnal periodic breathing , but it has been investigated almost exclusively in men . |

## 5. Scores with Background and Objective merged

### Full test set

Same predictions, scored again after treating Objective and Background as one class (4 classes). Nothing is retrained.

| Model | Sentences | 5-class accuracy | 5-class macro-F1 | Merged accuracy | Merged macro-F1 |
|---|---:|---:|---:|---:|---:|
| CNN | 29,578 | 86.4 | 80.9 | 90.7 | 90.6 |
| BiLSTM | 29,578 | 89.8 | 84.5 | 94.2 | 94.4 |

### LLM sample

| Model | Sentences | 5-class accuracy | 5-class macro-F1 | Merged accuracy | Merged macro-F1 |
|---|---:|---:|---:|---:|---:|
| CNN | 2,328 | 86.5 | 80.9 | 91.0 | 91.0 |
| BiLSTM | 2,328 | 88.8 | 83.1 | 93.9 | 94.3 |
| GPT sentence 0-shot | 2,328 | 90.0 | 84.1 | 95.3 | 95.7 |
| GPT abstract 3-shot | 2,328 | 92.0 | 85.8 | 97.3 | 97.7 |
