import os
import json
import numpy as np
import sys
import random
import openai
from tqdm import tqdm
import pandas as pd
import time
import re
import argparse

def get_system_prompt(args):
	if args.dataset == "bills" and args.label_categories == "broad":
		system_prompt = """You are a helpful research assistant with lots of knowledge about topic models. You are given a document assigned to a topic by a topic model. Annotate the document with a broad label, for example "health", "public lands", "domestic commerce", "government operations" and "defense".

Reply with a single word or phrase, indicating the label of the document."""

	elif args.dataset == "wikitext" and args.label_categories == "broad":
		system_prompt = """You are a helpful research assistant with lots of knowledge about topic models. You are given a document assigned to a topic by a topic model. Annotate the document with a broad label, for example csc"television", "songs", "transport", "warships and naval units", and "biology and medicine".

Reply with a single word or phrase, indicating the label of the document."""

	elif args.dataset == "wikitext" and args.label_categories == "specific":
		system_prompt = """You are a helpful research assistant with lots of knowledge about topic models. You are given a document assigned to a topic by a topic model. Annotate the document with a specific label, for example "tropical cyclones: atlantic", "actors, directors, models, performers, and celebrities", "road infrastructure: midwestern united states", "armies and military units", and "warships of germany".

Reply with a single word or phrase, indicating the label of the document."""
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

	dataset = "bills"
	if args.dataset == "bills":
		column = "summary"
	else:
		column ="text"

	with open(os.path.join("data", args.dataset, "processed/labeled/vocab_15k/vocab.json")) as f:
		vocab = json.load(f)

	vocab = {j:i for i,j in vocab.items()}

	# load metadata
	df_metadata = pd.read_json(os.path.join("data", args.dataset, "processed/labeled/vocab_15k/train.metadata.jsonl"), lines=True)

	paths = ["runs/outputs/k_selection/" + args.dataset + "-labeled/vocab_15k/k-" + str(i) for i in range(20, 420, 20)]

	output_path = "number-of-topics-section-4"
	os.makedirs(output_path, exist_ok=True)

	with open(os.path.join(output_path, "document_label_assignment_" + args.dataset + "_" + args.label_categories + ".jsonl"), "w") as outfile:
		for path in tqdm(paths):
			path = os.path.join(path, "2972")
			beta = np.load(os.path.join(path, "beta.npy"))
			theta = np.load(os.path.join(path, "train.theta.npy")).T # transpose
			
			print (beta.shape) # (20, 15'000), each row is probability distribution over vocab
			print (theta.shape) # (20, 32'661), each row is a probability distribution over documents?
			num_topics = beta.shape[0]

			# sample some topics
			sampled_topics = random.sample(list(range(num_topics)), 5)

			# for each topic
			for topic in sampled_topics:
				# sample top documents
				num_topics = 0
				user_prompt = ""
				arg_indices = np.argsort(theta[topic])[::-1][:10]

				for k, index in enumerate(arg_indices):
					# get text of this document
					text = df_metadata[column].iloc[index]
					text = " ".join(text.split()[:50]) # only take first 50 words
					user_prompt = text
					print (system_prompt)
					print (user_prompt)
					response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}], temperature=0.0, max_tokens=20)["choices"][0]["message"]["content"].strip()
					print ("topic", topic, "response --", response)
					out = {"path": path, "user_prompt": user_prompt, "response": response, "topic": topic, "k":k}
					json.dump(out, outfile)
					outfile.write("\n")
					time.sleep(0.1)
