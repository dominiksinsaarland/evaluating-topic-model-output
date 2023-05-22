import pandas as pd
import json
import random
import openai
from tqdm import tqdm
from ast import literal_eval
from collections import defaultdict
import time
import argparse
import os

def get_prompts(include_dataset_description="include"):
	if include_dataset_description == "include":
		system_prompt_NYT = """You are a helpful assistant evaluating the top words of a topic model output for a given topic. Select which word is the least related to all other words. If multiple words do not fit, choose the word that is most out of place. 
The topic modeling is based on The New York Times corpus. The corpus consists of articles from 1987 to 2007. Sections from a typical paper include International, National, New York Regional, Business, Technology, and Sports news; features on topics such as Dining, Movies, Travel, and Fashion; there are also obituaries and opinion pieces.
Reply with a single word."""

		system_prompt_wikitext = """You are a helpful assistant evaluating the top words of a topic model output for a given topic. Select which word is the least related to all other words. If multiple words do not fit, choose the word that is most out of place.
The topic modeling is based on the Wikipedia corpus. Wikipedia is an online encyclopedia covering a huge range of topics. Articles can include biographies ("George Washington"), scientific phenomena ("Solar Eclipse"), art pieces ("La Danse"), music ("Amazing Grace"), transportation ("U.S. Route 131"), sports ("1952 winter olympics"), historical events or periods ("Tang Dynasty"), media and pop culture ("The Simpsons Movie"), places ("Yosemite National Park"), plants and animals ("koala"), and warfare ("USS Nevada (BB-36)"), among others.
Reply with a single word."""
		outfile_name = "coherence-outputs-section-2/intrusion_outfile_with_dataset_description.jsonl"
	else:
		system_prompt_NYT = """You are a helpful assistant evaluating the top words of a topic model output for a given topic. Select which word is the least related to all other words. If multiple words do not fit, choose the word that is most out of place. 
Reply with a single word."""

		system_prompt_wikitext = """You are a helpful assistant evaluating the top words of a topic model output for a given topic. Select which word is the least related to all other words. If multiple words do not fit, choose the word that is most out of place.
Reply with a single word."""
		outfile_name = "coherence-outputs-section-2/intrusion_outfile_without_dataset_description.jsonl"
	return 	system_prompt_NYT, system_prompt_wikitext, outfile_name

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--API_KEY", default="openai API key", type=str, required=True, help="valid openAI API key")
	parser.add_argument("--dataset_description", default="include", type=str, help="whether to include a dataset description or not, default = include.")
	args = parser.parse_args()


	openai.api_key = args.API_KEY
	random.seed(42)
	df = pd.read_json("intruder_outfile.jsonl", lines=True)

	system_prompt_NYT, system_prompt_wikitext, outfile_name = get_prompts(include_dataset_description=args.dataset_description)
	os.makedirs("coherence-outputs-section-2", exist_ok=True)


	columns = df.columns.tolist()

	with open(outfile_name, "w") as outfile:
		for i, row in tqdm(df.iterrows(), total=len(df)):
			if row.dataset_name == "wikitext":
				system_prompt = system_prompt_wikitext
			else:
				system_prompt = system_prompt_NYT

			words = row.topic_terms
			# shuffle words
			random.shuffle(words)
			# we only prompt 5 words
			words = words[:5]
			
			# we add intruder term
			intruder_term = row.intruder_term
			
			# we shuffle again
			words.append(intruder_term)
			random.shuffle(words)	
			
			# we have a user prompt			
			user_prompt = ", ".join(['"' + w + '"' for w in words])

			out = {col: row[col] for col in columns}
			response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}], temperature=0.7, max_tokens=15)["choices"][0]["message"]["content"].strip()
			out["response"] = response
			out["user_promt"] = user_prompt
			json.dump(out, outfile)
			outfile.write("\n")
			time.sleep(0.5)


