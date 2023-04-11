
<div align="center">

# Text Similarity as An Evaluation Measure of Text Generation

[![Code style - black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
</div>

## ❓ Context

Natural Language Generation (NLG) is the process of generating human-like language by machines. One of the key challenges in evaluating the quality of generated text is to compare it with 'gold standard' references.

However, obtaining human annotations for evaluation is an expensive and time-consuming process, making it impractical for large-scale experiments. As a result, researchers have explored alternative methods for evaluating the quality of generated text.

Two families of metrics have been proposed: trained metrics and untrained metrics. While trained metrics may not generalize well to new data, untrained metrics, such as word or character-based metrics and embedding-based metrics, offer a more flexible and cost-effective solution. To assess the performance of an evaluation metric, correlation measures such as Pearson, Spearman, or Kendall tests are used, either at the text-level or system-level.

## 🎯 Objective

This project aims to benchmark the correlation of existing metrics with human scores on generation task: translation or data2text generation or story generation.

## :rocket: How to use the project

1. First, you need to clone the repository and `cd` into it :
```bash
git clone
cd NLG_EVALUATION_METRICS
```
2. Then, you need to create a virtual environment and activate it :
```bash
python3 -m venv venv
source venv/bin/activate
```
3. You need to install all the `requirements` using the following command :
```bash
pip install -r requirements.txt
```
4. [Optional] if you are using this repository in development mode, you can run the following command to set up the git hook scripts:
```bash
pre-commit install
```
4. You can now run the python files in the `file name` folder using the following commands :
```bash
cd cluster
python3 [file name]
```

To test the project, you can run the `test.ipynb` notebook.

## :memo: Results
TODO: List the tested metrics

TODO: Describe the criteria used to evaluate the results

| Metric | criterion1 | criterion2 | criterion3 |
|---|---|---|---|
| TER | XX | XX | XX |
| DepthScore | XX | XX | XX |

TODO: Describe and analyse the results

## 🤔 What's next ?
TODO: List the next steps

## :books: References
TODO: List the references

## :pencil2: Authors
- LETAIEF Maram
- LIDAM Soukaina
