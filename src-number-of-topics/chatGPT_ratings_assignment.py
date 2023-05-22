import os
import json
import numpy as np
import sys
import random
import openai
from tqdm import tqdm
import pandas as pd
import argparse
import time


def get_system_prompt(args):
	if args.dataset == "bills" and args.label_categories == "broad":
		system_prompt = """You are a helpful assistant evaluating the top words of a topic model output for a given topic. Please rate how related the following words are to each other on a scale from 1 to 3 ("1" = not very related, "2" = moderately related, "3" = very related). 
The topic modeling is based on a legislative Bill summary dataset. We are interested in coherent broad topics. Typical topics in the dataset include "Health", "Public Lands", "Domestic Commerce", "Government Operations", or "Defense".
Reply with a single number, indicating the overall appropriateness of the topic."""

	elif args.dataset == "wikitext" and args.label_categories == "broad":
		system_prompt = """You are a helpful assistant evaluating the top words of a topic model output for a given topic. Please rate how related the following words are to each other on a scale from 1 to 3 ("1" = not very related, "2" = moderately related, "3" = very related). 
		The topic modeling is based on the Wikipedia corpus. Wikipedia is an online encyclopedia covering a huge range of topics. Typical topics in the dataset include "television", "songs", "transport", "warships and naval units", and "biology and medicine".
		Reply with a single number, indicating the overall appropriateness of the topic."""

	elif args.dataset == "wikitext" and args.label_categories == "specific":
		system_prompt = """You are a helpful assistant evaluating the top words of a topic model output for a given topic. Please rate how related the following words are to each other on a scale from 1 to 3 ("1" = not very related, "2" = moderately related, "3" = very related). 
The topic modeling is based on the Wikipedia corpus. Wikipedia is an online encyclopedia covering a huge range of topics. Typical topics in the dataset include "tropical cyclones: atlantic", "actors, directors, models, performers, and celebrities", "road infrastructure: midwestern united states", "armies and military units", and "warships of germany".
Reply with a single number, indicating the overall appropriateness of the topic."""
	else:
		print ("experiment not implemented")
		sys.exit(0)
	return system_prompt

# add logit bias
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--API_KEY", default="openai API key", type=str, required=True, help="valid openAI API key")
	parser.add_argument("--dataset", default="wikitext", type=str, help="dataset (wikitext or bills)")
	parser.add_argument("--label_categories", default="broad", type=str, help="granularity of ground-truth labels (part of the prompt): broad or specific.")
	args = parser.parse_args()

	random.seed(42)
	system_prompt = get_system_prompt(args)
	openai.api_key = args.API_KEY

	with open(os.path.join("data", args.dataset, "processed/labeled/vocab_15k/vocab.json")) as f:
		vocab = json.load(f)
	vocab = {j:i for i,j in vocab.items()}

	paths = ["runs/outputs/k_selection/" + args.dataset + "-labeled/vocab_15k/k-" + str(i) for i in range(20, 420, 20)]


	output_path = "number-of-topics-section-4"
	os.makedirs(output_path, exist_ok=True)

	with open(os.path.join(output_path, "coherence_ratings_" + args.dataset + "_" + args.label_categories + ".jsonl"), "w") as outfile:
		for path in tqdm(paths):
			path = os.path.join(path, "2972")
			beta = np.load(os.path.join(path, "beta.npy"))
			theta = np.load(os.path.join(path, "train.theta.npy"))
			
			print (beta.shape) # 20, 15'000, each row is probability distribution over vocab
			print (theta.shape)
			num_topics = beta.shape[0]
			top_words = []
			for row in beta: 
				indices = row.argsort()[::-1][:10]
				top_topic_words = [vocab[i] for i in indices]
				top_words.append(top_topic_words)

			# sample 10 topics

			sampled_topics = random.sample(list(range(num_topics)), k=10)
			for i in sampled_topics:
				topic = top_words[i]
				random.shuffle(topic)
				user_prompt = ", ".join(topic)

				response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}], temperature=0, max_tokens=1, logit_bias={16:100, 17:100, 18:100})["choices"][0]["message"]["content"].strip()
				out = {"path": path, "topic": i, "user_prompt": user_prompt, "response": response}
				json.dump(out, outfile)
				outfile.write("\n")
				print (response)
				time.sleep(0.1)
