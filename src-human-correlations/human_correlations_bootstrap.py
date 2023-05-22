import pandas as pd
import re
from scipy.stats import spearmanr
import numpy as np
from sklearn.metrics import accuracy_score
import random, json
from ast import literal_eval
from tqdm import tqdm
import os
import argparse

def load_dataframe(fn, task=""):
	if fn.endswith(".csv"):
		df = pd.read_csv(fn)
	elif fn.endswith("jsonl"):
		df = pd.read_json(fn, lines=True)
	print (df)
	print (df.iloc[0])
	if task == "intrusion":
		if "response_correct" not in df.columns:
			df["response_correct"] = postprocess_chatGPT_intrusion_test(df)
	else:

		if "gpt_ratings" not in df.columns:
			df["gpt_ratings"] = df.response.astype(int)		
	return df
	
def postprocess_chatGPT_intrusion_test(df):
	response = df.response.tolist()
	response = [i.lower() for i in response]
	response = [re.findall(r"\b[a-z_\d]+\b", i) for i in response]
	response_correct = []	
	#df.topic_terms = df.topic_terms.apply(lambda x: literal_eval(x))
	#for i,j,k in zip(response, df.intruder_term, df.topic_terms):
	for i,j in zip(response, df.intruder_term):
		if not i:
			response_correct.append(0)
		elif i[0] == j:
			response_correct.append(1)
		else:
			response_correct.append(0)
	return response_correct		

def postprocess_chatGPT_intrusion_new(df):
	for response, intruder_term, topic_words in zip(response, df.intruder_term, df.topic_terms):
		if not response:
			response_correct.append(0)
		elif response.lower() == intruder_term.lower():
			response_correct.append(1)
		else:
			response_correct.append(0)
	return response_correct		

def get_confidence_intervals(statistic):
	statistic = sorted(statistic)
	lower = statistic[4] # it's the fourth lowest value; if we had 100 samples, it would be the 2.5nd lowest value, this * 1.5 gets us 3.75
	upper = statistic[-4] # it's the fourth highest value
	print ("lower", lower, "upper", upper)
	
def get_filenames(with_dataset_description = True):
	if with_dataset_description:
		intrusion_fn = "coherence-outputs-section-2/intrusion_outfile_with_dataset_description.jsonl"
		rating_fn = "coherence-outputs-section-2/ratings_outfile_with_dataset_description.csv"
	else:
		intrusion_fn = "coherence-outputs-section-2/intrusion_outfile_without_dataset_description.csv"
		rating_fn = "coherence-outputs-section-2/ratings_outfile_without_dataset_description.jsonl"
	return intrusion_fn, rating_fn


def compute_human_ceiling_intrusion(data, only_confident = False):
	ratings_human, ratings_chatGPT = [], []
	spearman_wiki = []
	spearman_NYT = []
	spearman_concat = []
	for _ in tqdm(range(1000), total=1000):
		ratings_human, ratings_chatGPT = [], []
		for dataset in ["wikitext", "nytimes"]:
			for model in ["mallet", "dvae", "etm"]:
				for topic_id in range(50):
					intrusion_scores_raw = data[dataset][model]["metrics"]["intrusion_scores_raw"][topic_id]
					if only_confident:
						intrusion_scores_raw = [i for i,j in zip(intrusion_scores_raw, data[dataset][model]["metrics"]["intrusion_confidences_raw"][topic_id]) if j == 1]
					if not intrusion_scores_raw:
						intrusion_scores_raw = [0]

					if len(intrusion_scores_raw) == 1:
						intrusion_scores_1 = intrusion_scores_raw
						intrusion_scores_2 = intrusion_scores_raw
					else:
						length = len(intrusion_scores_raw) // 2
						intrusion_scores_1 = intrusion_scores_raw[:length]
						intrusion_scores_2 = intrusion_scores_raw[length:]
					intrusion_scores_1 = random.choices(intrusion_scores_1, k=len(intrusion_scores_1))
					intrusion_scores_2 = random.choices(intrusion_scores_2, k=len(intrusion_scores_2))
					ratings_human.append(np.mean(intrusion_scores_1))
					ratings_chatGPT.append(np.mean(intrusion_scores_2))
		spearman_wiki.append(spearmanr(ratings_chatGPT[:150], ratings_human[:150]).statistic)
		spearman_NYT.append(spearmanr(ratings_chatGPT[150:], ratings_human[150:]).statistic)
		spearman_concat.append(spearmanr(ratings_chatGPT, ratings_human).statistic)
	print ("wiki", np.mean(spearman_wiki))
	#get_confidence_intervals(spearman_wiki)
	print ("NYT",  np.mean(spearman_NYT))
	#get_confidence_intervals(spearman_NYT)
	print ("concat", np.mean(spearman_concat))
	#get_confidence_intervals(spearman_concat)

def compute_human_ceiling_rating(data, only_confident = False):
	ratings_human, ratings_chatGPT = [], []
	spearman_wiki = []
	spearman_NYT = []
	spearman_concat = []
	for _ in tqdm(range(1000), total=1000):
		ratings_human, ratings_chatGPT = [], []
		for dataset in ["wikitext", "nytimes"]:
			for model in ["mallet", "dvae", "etm"]:
				for topic_id in range(50):
					intrusion_scores_raw = data[dataset][model]["metrics"]["ratings_scores_raw"][topic_id]
					if only_confident:
						intrusion_scores_raw = [i for i,j in zip(intrusion_scores_raw, data[dataset][model]["metrics"]["ratings_confidences_raw"][topic_id]) if j == 1]
						
					if not intrusion_scores_raw:
						intrusion_scores_raw = [1]
					if len(intrusion_scores_raw) == 1:
						intrusion_scores_1 = intrusion_scores_raw
						intrusion_scores_2 = intrusion_scores_raw
					else:
						length = len(intrusion_scores_raw) // 2
						intrusion_scores_1 = intrusion_scores_raw[:length]
						intrusion_scores_2 = intrusion_scores_raw[length:]
					intrusion_scores_1 = random.choices(intrusion_scores_1, k=len(intrusion_scores_1))
					intrusion_scores_2 = random.choices(intrusion_scores_2, k=len(intrusion_scores_2))
					ratings_human.append(np.mean(intrusion_scores_1))
					ratings_chatGPT.append(np.mean(intrusion_scores_2))
		spearman_wiki.append(spearmanr(ratings_chatGPT[:150], ratings_human[:150]).statistic)
		spearman_NYT.append(spearmanr(ratings_chatGPT[150:], ratings_human[150:]).statistic)
		spearman_concat.append(spearmanr(ratings_chatGPT, ratings_human).statistic)
	print ("wiki", np.mean(spearman_wiki))
	#get_confidence_intervals(spearman_wiki)
	print ("NYT",  np.mean(spearman_NYT))
	#get_confidence_intervals(spearman_NYT)
	print ("concat", np.mean(spearman_concat))
	#get_confidence_intervals(spearman_concat)
	

def compute_spearmanr_bootstrap_intrusion(df_intruder_scores, data, only_confident=False):
	ratings_human, ratings_chatGPT = [], []
	spearman_wiki = []
	spearman_NYT = []
	spearman_concat = []

	for _ in tqdm(range(1000), total=1000):
		ratings_human, ratings_chatGPT = [], []
		for dataset in ["wikitext", "nytimes"]:
			for model in ["mallet", "dvae", "etm"]:
				for topic_id in range(50):
					intrusion_scores_raw = data[dataset][model]["metrics"]["intrusion_scores_raw"][topic_id]
					if only_confident:
						intrusion_scores_raw = [i for i,j in zip(intrusion_scores_raw, data[dataset][model]["metrics"]["intrusion_confidences_raw"][topic_id]) if j == 1]
					# sample bootstrap fold
					
					intrusion_scores_raw = random.choices(intrusion_scores_raw, k=len(intrusion_scores_raw))
					df_topic = df_intruder_scores[(df_intruder_scores.dataset_name == dataset) & (df_intruder_scores.model_name == model) & (df_intruder_scores.topic_id == topic_id)]
					gpt_ratings = random.choices(df_topic.response_correct.tolist(), k= len(df_topic.response_correct))
					
					# save results
					ratings_human.append(np.mean(intrusion_scores_raw))
					ratings_chatGPT.append(np.mean(gpt_ratings))
			# compute spearman_R and save results
		spearman_wiki.append(spearmanr(ratings_chatGPT[:150], ratings_human[:150]).statistic)
		spearman_NYT.append(spearmanr(ratings_chatGPT[150:], ratings_human[150:]).statistic)
		spearman_concat.append(spearmanr(ratings_chatGPT, ratings_human).statistic)
	print ("wiki", np.mean(spearman_wiki))
	get_confidence_intervals(spearman_wiki)
	print ("NYT",  np.mean(spearman_NYT))
	get_confidence_intervals(spearman_NYT)
	print ("concat", np.mean(spearman_concat))
	get_confidence_intervals(spearman_concat)

def compute_spearmanr_bootstrap_rating(df_rating_scores, data, only_confident=False):
	ratings_human, ratings_chatGPT = [], []
	spearman_wiki = []
	spearman_NYT = []
	spearman_concat = []

	for _ in tqdm(range(150), total=150):
		ratings_human, ratings_chatGPT = [], []
		for dataset in ["wikitext", "nytimes"]:
			for model in ["mallet", "dvae", "etm"]:
				for topic_id in range(50):
					rating_scores_raw = data[dataset][model]["metrics"]["ratings_scores_raw"][topic_id]
					if only_confident:
						rating_scores_raw = [i for i,j in zip(rating_scores_raw, data[dataset][model]["metrics"]["ratings_confidences_raw"][topic_id]) if j == 1]
					# sample bootstrap fold
					rating_scores_raw = random.choices(rating_scores_raw, k=len(rating_scores_raw))
					df_topic = df_rating_scores[(df_rating_scores.dataset_name == dataset) & (df_rating_scores.model_name == model) & (df_rating_scores.topic_id == topic_id)]
					gpt_ratings = random.choices(df_topic.gpt_ratings.tolist(), k= len(df_topic.gpt_ratings))
					
					# save results
					ratings_human.append(np.mean(rating_scores_raw))
					ratings_chatGPT.append(np.mean(gpt_ratings))
		# compute spearman_R and save results
		spearman_wiki.append(spearmanr(ratings_chatGPT[:150], ratings_human[:150]).statistic)
		spearman_NYT.append(spearmanr(ratings_chatGPT[150:], ratings_human[150:]).statistic)
		spearman_concat.append(spearmanr(ratings_chatGPT, ratings_human).statistic)
	print ("wiki", np.mean(spearman_wiki))
	get_confidence_intervals(spearman_wiki)
	print ("NYT",  np.mean(spearman_NYT))
	get_confidence_intervals(spearman_NYT)
	print ("concat", np.mean(spearman_concat))
	get_confidence_intervals(spearman_concat)




if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--task", default="ratings", type=str)
	parser.add_argument("--filename", default="coherence-outputs-section-2/ratings_outfile_with_dataset_description.jsonl", type=str, help="whether to include a dataset description or not, default = include.")
	parser.add_argument("--only_confident", default="true", type=str)
	args = parser.parse_args()
	
	if args.only_confident == "true":
		only_confident = True
	else:
		only_confident = False

	random.seed(42)
	path = "coherence-outputs-section-2"
	
	experiments = ["human_ceiling", "dataset_description", "dataset_description_only_confident", "no_dataset_description"]
	
	with open("all_data.json") as f:
		data = json.load(f)

	if args.task == "human_ceiling":
		compute_human_ceiling_intrusion(data, only_confident=only_confident)
		compute_human_ceiling_rating(data, only_confident=only_confident)
	elif args.task == "ratings":
		df_rating = load_dataframe(args.filename)
		compute_spearmanr_bootstrap_rating(df_rating, data, only_confident=only_confident)			
		
	elif args.task == "intrusion":
		df_rating = load_dataframe(args.filename)
		compute_spearmanr_bootstrap_rating(df_rating, data, only_confident=only_confident)			
