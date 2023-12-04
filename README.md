# Re-visiting Automated Topic Model Evaluation with Large Language Models

This repo contains code and data for our [EMNLP 2023 paper](https://aps.arxiv.org/abs/2305.12152) about assessing topic model output with Large Language Models.
```
@misc{stammbach2023revisiting,
  title={Re-visiting Automated Topic Model Evaluation with Large Language Models}, 
  author={Dominik Stammbach and Vilém Zouhar and Alexander Hoyle and Mrinmaya Sachan and Elliott Ash},
  year={2023},
  url={https://aps.arxiv.org/abs/2305.12152},
  eprint={2305.12152},
  archivePrefix={arXiv},
  primaryClass={cs.CL}
}
```

## Prerequisites

```shell
pip install --upgrade openai pandas
```

## Large Language Models and Topics with Human Annotations

Download topic words and human annotations from the paper [Is Automated Topic Model Evaluation Broken?](https://arxiv.org/abs/2107.02173) from their [github repository](https://github.com/ahoho/topics/blob/dev/data/human/all_data/all_data.json).


### Intruder Detection Test

Following (Hoyle et al., 2021), we randomly sample intruder words which are not in the top 50 topic words for each topic.

```shell
python src-human-correlations/generate_intruder_words_dataset.py 
```

We can then call an LLM to automatically annotate the intruder words for each topic. 

```shell
python src-human-correlations/chatGPT_evaluate_intruders.py --API_KEY a_valid_openAI_api_key
```

For the ratings task, simply call the file which rates topic word sets (no need to generate a dataset first)

```shell
python src-human-correlations/chatGPT_evaluate_topic_ratings.py --API_KEY a_valid_openAI_api_key
```
(In case the openAI API breaks, we simply save all output in a json file, and would restart the script while skipping all already annotated datapoints.)


### Evaluation LLMs and Human Correlations

We evaluate using a bootstrapp appraoch where we sample human annotations and LLM annotations for each datapoint. We then average these sampled annotation, and compute a spearman's rho for each bootstrapped sample. We report the mean spearman's rho over all 1000 bootstrapped samples.

```shell
python src-human-correlations/human_correlations_bootstrap.py --filename coherence-outputs-section-2/ratings_outfile_with_dataset_description.jsonl --task ratings
```


## Evaluating Topic Models with Different Numbers of Topics

Download fitted topic models and metadata for two datasets (bills and wikitext) [here](https://www.dropbox.com/s/huxdloe5l6w2tu5/topic_model_k_selection.zip?dl=0) and unzip

### Rating Topic Word Sets

To run LLM ratings of topic word sets on a dataset (wiki or NYT) with broad or specific ground-truth example topics, simply run:

```shell
python src-number-of-topics/chatGPT_ratings_assignment.py --API_KEY a_valid_openAI_api_key --dataset wikitext --label_categories broad
```

### Purity of Document Collections

We also assign a document label to the top documents belonging to a topic, following [Doogan and Buntine, 2021](https://aclanthology.org/2021.naacl-main.300/). We then average purity per document collection, and the number of topics with on averag highest purities is the preferred cluster of this procedure.

To run LLM label assignments on a dataset (wiki or NYT) with broad or specific ground-truth example topics, simply run:

```shell
python src-number-of-topics/chatGPT_document_label_assignment.py --API_KEY a_valid_openAI_api_key --dataset wikitext --label_categories broad
```

### Plot resulting scores

```shell
python src-number-of-topics/LLM_scores_and_ARI.py --label_categories broad --method label_assignment --dataset bills --label_categories broad --filename number-of-topics-section-4/document_label_assignment_wikitext_broad.jsonl
```

## Questions

Please contact [Dominik Stammbach](mailto:dominik.stammbach@gess.ethz.ch) regarding any questions.


[![Paper video presentation](https://img.youtube.com/vi/qIDjtgWTgjs/0.jpg)](https://www.youtube.com/watch?v=qIDjtgWTgjs)
