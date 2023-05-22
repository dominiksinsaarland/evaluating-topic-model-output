import json
import random
import openai
from tqdm import tqdm
import pandas as pd
import time
import argparse
import os


def get_prompts(include_dataset_description=True):
	if include_dataset_description:
		system_prompt_NYT = """You are a helpful assistant evaluating the top words of a topic model output for a given topic. Please rate how related the following words are to each other on a scale from 1 to 3 ("1" = not very related, "2" = moderately related, "3" = very related). 
The topic modeling is based on The New York Times corpus. The corpus consists of articles from 1987 to 2007. Sections from a typical paper include International, National, New York Regional, Business, Technology, and Sports news; features on topics such as Dining, Movies, Travel, and Fashion; there are also obituaries and opinion pieces.
Reply with a single number, indicating the overall appropriateness of the topic."""

		system_prompt_wikitext = """You are a helpful assistant evaluating the top words of a topic model output for a given topic. Please rate how related the following words are to each other on a scale from 1 to 3 ("1" = not very related, "2" = moderately related, "3" = very related). 
The topic modeling is based on the Wikipedia corpus. Wikipedia is an online encyclopedia covering a huge range of topics. Articles can include biographies ("George Washington"), scientific phenomena ("Solar Eclipse"), art pieces ("La Danse"), music ("Amazing Grace"), transportation ("U.S. Route 131"), sports ("1952 winter olympics"), historical events or periods ("Tang Dynasty"), media and pop culture ("The Simpsons Movie"), places ("Yosemite National Park"), plants and animals ("koala"), and warfare ("USS Nevada (BB-36)"), among others.
Reply with a single number, indicating the overall appropriateness of the topic."""
		outfile_name = "coherence-outputs-section-2/ratings_outfile_with_dataset_description.jsonl"
	else:
		system_prompt_NYT = """You are a helpful assistant evaluating the top words of a topic model output for a given topic. Please rate how related the following words are to each other on a scale from 1 to 3 ("1" = not very related, "2" = moderately related, "3" = very related). 
Reply with a single number, indicating the overall appropriateness of the topic."""

		system_prompt_wikitext = """You are a helpful assistant evaluating the top words of a topic model output for a given topic. Please rate how related the following words are to each other on a scale from 1 to 3 ("1" = not very related, "2" = moderately related, "3" = very related). 
Reply with a single number, indicating the overall appropriateness of the topic."""
		outfile_name = "coherence-outputs-section-2/ratings_outfile_without_dataset_description.jsonl"
	return 	system_prompt_NYT, system_prompt_wikitext, outfile_name


if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("--API_KEY", default="openai API key", type=str, required=True, help="valid openAI API key")
	parser.add_argument("--dataset_description", default="include", type=str, help="whether to include a dataset description or not, default = include.")
	args = parser.parse_args()

	openai.api_key = args.API_KEY
	random.seed(42)

	with open("all_data.json") as f:
		data = json.load(f)
	random.seed(42)

	system_prompt_NYT, system_prompt_wikitext, outfile_name = get_prompts(include_dataset_description=args.dataset_description)	
	os.makedirs("coherence-outputs-section-2", exist_ok=True)

	with open(outfile_name, "w") as outfile:
		for dataset_name, dataset in data.items():
			for model_name, dataset_model in dataset.items():
				print (dataset_name, model_name)
				topics = dataset_model["topics"]
				human_evaluations = dataset_model["metrics"]["ratings_scores_avg"]
				i = 0
				for topic, human_eval in tqdm(zip(topics, human_evaluations), total=50):
					topic = topic[:10]
					for run in range(3):
						random.shuffle(topic)
						user_prompt = ", ".join(topic)
						if dataset_name == "wikitext":
							system_prompt = system_prompt_wikitext
						else:
							system_prompt = system_prompt_NYT
						response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}], temperature=1.0, logit_bias={16:100, 17:100, 18:100}, max_tokens=1)["choices"][0]["message"]["content"].strip()
						out = {"dataset_name": dataset_name, "model_name": model_name, "topic_id": i, "user_prompt": user_prompt, "response": response, "human_eval":human_eval, "run": run}
						json.dump(out, outfile)
						outfile.write("\n")
						time.sleep(0.5)
					i += 1

