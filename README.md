SelfCheckGPT
=====================================================
Project page for our paper "[SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models](https://arxiv.org/abs/2303.08896)"

## Code/Package

### Installation

    pip install selfcheckgpt

### Example

See more details in Jupyter Notebook ```demo/demo1.ipynb```

```python
from selfcheckgpt.modeling_mqag import SelfCheckMQAG
from selfcheckgpt.modeling_bertscore import SelfCheckBERTScore

selfcheck_mqag = SelfCheckMQAG()
selfcheck_bertscore = SelfCheckBERTScore()

sent_scores_mqag = selfcheck_mqag.predict(
    sentences,
    passage,
    [sample1, sample2, sample3],
    num_questions_per_sent = 5,
    scoring_method = 'bayes_with_alpha',
    beta1 = 0.8, beta2 = 0.8,
)
sent_scores_bertscore = selfcheck_bertscore.predict(
    sentences,
    [sample1, sample2, sample3],
)

```

## Dataset
The `wiki_bio_gpt3_hallucination` dataset currently consists of 65 annotated passages. You can find more information in the paper or our data card on HuggingFace: https://huggingface.co/datasets/potsawee/wiki_bio_gpt3_hallucination. To use this dataset, you can either load it through HuggingFace dataset API, or download it directly from below in the JSON format.

### Option1: HuggingFace

```python
from datasets import load_dataset
dataset = load_dataset("potsawee/wiki_bio_gpt3_hallucination")
```

### Option2: Manual Download
Download from our [Google Drive](https://drive.google.com/drive/folders/1-45BC-AZQQtxIuaWSFcT-XQ-nUvq0zZB?usp=sharing), then you can load it in python:

```python
import json
with open("dataset.json", "r") as f:
    content = f.read()
dataset = json.loads(content)
```

Each instance consists of:
- `gpt3_text`: GPT-3 generated passage
- `wiki_bio_text`: Actual Wikipedia passage (first paragraph)
- `gpt3_sentences`: `gpt3_text` split into sentences using `spacy`
- `annotation`: human annotation at the sentence level
-  `wiki_bio_test_idx`: ID of the concept/individual from the original wikibio dataset (testset)

## Citation

```
@misc{manakul2023selfcheckgpt,
      title={SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models},
      author={Potsawee Manakul and Adian Liusie and Mark J. F. Gales},
      year={2023},
      eprint={2303.08896},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
