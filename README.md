# Challenges in Measuring Bias via Open-Ended Language Generation

Requirements
```
pandas
scipy
openai (for GPT-3 experiments)
vaderSentiment (sentiment)
detoxify (for unitaryai toxicity scores)
```

Clone below repositories under this repo:
```
https://github.com/ewsheng/nlg-bias (Required for regard computations.)
https://github.com/MilaNLProc/honest 
https://github.com/amazon-research/bold (Create a folder data/ and place bold under data.)
```

Standard pipeline:
Scripts are available under `scripts/`. We first complete prompts by calling `complete_prompts.py` and then compute defining metrics using `compute_tox_sent.py`. In the latter, comment out the metrics that you are not interested in which is important because computing regard requires a different running environment.

Other requirements:
- To create completions given prompts, download respective models from huggingface/transformers e.g.
```
git lfs install
git clone https://huggingface.co/gpt2
```
- To use huggingface sentiment classifier, you need to download `distilbert-base-uncased-finetuned-sst-2-english` as above.
- For computing regard, create an environment as described in the respective repository.
- You will need a file `openai_key` including your OPEN AI API key as one-liner in order to run GPT-3 experiments.
