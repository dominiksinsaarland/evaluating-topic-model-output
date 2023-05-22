import matplotlib.pyplot as plt
import pandas as pd
import re, os, json, random
from tqdm import tqdm
import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_mutual_info_score, adjusted_rand_score, completeness_score, homogeneity_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
from collections import defaultdict
import argparse
from collections import Counter

def moving_average(a, window_size=3):
	if window_size == 0:
		return a
	out = []
	for i in range(len(a)):
		start = max(0, i - window_size)
		out.append(np.mean(a[start:i + 1]))
	return out

def postprocess_labels(df):
	import spacy
	nlp = spacy.load("en_core_web_sm")
	if "response" not in df.columns:
		df["response"] = df.chatGPT_eval.tolist()
	out = []

	for i, text in enumerate(df.response):
		text = text.lower()
		text = re.findall("[a-z ]+", text)[0] # oh, what would this do?
		text = " ".join([w.lemma_ for w in nlp(text)])
		out.append(text)
	df["response"] = out
	return df

def compute_ARI(args):
	df_metadata = pd.read_json(os.path.join("data", args.dataset, "processed/labeled/vocab_15k/train.metadata.jsonl"), lines=True)
	paths = ["runs/outputs/k_selection/" + args.dataset + "-labeled/vocab_15k/k-" + str(i) for i in range(20, 420, 20)]
	
	if args.dataset == "bills":
		topic = df_metadata.topic.tolist()
	else:
		if args.label_categories == "broad":
			topic = df_metadata.category.tolist()
		else:
			topic = df_metadata.subcategory.tolist()

	cluster_metrics = defaultdict(list)	
	for path,num_topics in tqdm(zip(paths, range(20,420,20)), total=20):
		path = os.path.join(path, "2972")
		beta = np.load(os.path.join(path, "beta.npy"))
		theta = np.load(os.path.join(path, "train.theta.npy"))
		argmax_theta = theta.argmax(axis=-1)
		cluster_metrics["ami"].append(adjusted_mutual_info_score(topic, argmax_theta))
		cluster_metrics["ari"].append(adjusted_rand_score(topic, argmax_theta))
		cluster_metrics["completeness"].append(completeness_score(topic, argmax_theta))
		cluster_metrics["homogeneity"].append(homogeneity_score(topic, argmax_theta))
	return cluster_metrics

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--dataset", default="wikitext", type=str, help="dataset (wikitext or bills)")
	parser.add_argument("--label_categories", default="broad", type=str, help="granularity of ground-truth labels (part of the prompt): broad or specific.")
	parser.add_argument("--filename", default="number-of-topics-section-4/document_label_assignment_wikitext_broad.sonl", type=str, help="filename with LLM responses")
	parser.add_argument("--method", default="label_assignment", type=str, help="if we use topic word set ratings or document label assignment (label_assignment | topic_ratings")
	
	args = parser.parse_args()

	cluster_metrics = compute_ARI(args)
	paths = ["../runs/outputs/k_selection/" + args.dataset + "-labeled/vocab_15k/k-" + str(i) for i in range(20, 420, 20)]
	df = pd.read_json(args.filename, lines=True)
	outfile_name = "n_clusters_" + args.dataset + "_" + args.label_categories + ".png"
	plt_label = "LLM Scores and ARI"


	average_goodness = []
	if args.method == "topic_ratings":
		df["gpt_ratings"] = df.response.astype(int)
		# get average gpt_ratings for each k
		for path in paths:
			path = os.path.join(path, "2972")
			df_at_k = df[df.path == path]
			average_goodness.append(df_at_k.gpt_ratings.mean())	
	elif args.method == "label_assignment":
		df = postprocess_labels(df)
		# get average purity for each k
		for path in paths:
			path = os.path.join(path, "2972")
			df_at_k = df[df.path == path]
			purities = []
			for topic in df_at_k.topic.unique():
				df_topic = df_at_k[df_at_k.topic == topic]
				labels = df_topic.response.tolist()
				most_common,num_most_common = Counter(labels).most_common(1)[0]
				purity = num_most_common / len(labels)
				purities.append(purity)
			average_goodness.append(np.mean(purities))

	average_goodness = moving_average(average_goodness) # smooth via moving_average to remove weird outliers
	ARI = cluster_metrics["ari"]

	fig = plt.figure()
	ax = fig.add_subplot(111)

	ax1 = plt.subplot()
	l1, = ax1.plot(average_goodness, color='tab:red')
	ax2 = ax1.twinx()
	l2, = ax2.plot(ARI, color='tab:blue')

	spearman_rho = spearmanr(average_goodness, ARI).statistic
	print (spearman_rho)

	plt.legend([l1, l2], ["LLM Score", "ARI"])
	ax.set_xlabel("Number of Topics")

	n_clusters = list(range(20, 420, 20)) 
	plt.xticks(range(len(n_clusters)), n_clusters, rotation=45)

	if len(n_clusters) > 12:
		every_nth = len(n_clusters) // 8
		for n, label in enumerate(ax.xaxis.get_ticklabels()):
			if n % every_nth != 0:
				label.set_visible(False)
	fig_title = "LLM Scores and ARI, " + args.dataset + ", " + args.label_categories + "$, \\rho = " + str(spearman_rho) + "$"
	plt.title(fig_title)
	plt.savefig(outfile_name)
