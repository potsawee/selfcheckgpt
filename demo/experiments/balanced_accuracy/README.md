Balanced Accuracy
===================================

- Sentence-level balanced accuracy for the Wikibio dataset
- For these results, the threshold is set to be 0.5
- As the metric is balanced accuracy, NonFact and Factual scenarios yield the same results

| Method               |  NonFact   |  NonFact*  |
|----------------------|:------------------:|:------------------:|
| Random Guessing      |        50.00       |        50.00       |
| SelfCheck-QA         |        62.87       |        60.08       |
| SelfCheck-NLI        |        70.55       |        62.15       |
| SelfCheck-Prompt (gpt-3.5-turbo) |      76.69     |     65.93      |

