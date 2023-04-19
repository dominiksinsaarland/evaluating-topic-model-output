import matplotlib.pyplot as plt
import pandas as pd
import re, os, json, random
from tqdm import tqdm
import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_mutual_info_score, adjusted_rand_score, completeness_score, homogeneity_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
from collections import defaultdict

def moving_average(a, window_size=3):
	if window_size == 0:
		return a
	out = []
	for i in range(len(a)):
		start = max(0, i - window_size)
		out.append(np.mean(a[start:i + 1]))
	return out

def compute_adjusted_NMI_bills():
	df_metadata = pd.read_json("data/bills/processed/labeled/vocab_15k/train.metadata.jsonl", lines=True)
	print (df_metadata)
	paths = ["runs/outputs/k_selection/" + "bills" + "-labeled/vocab_15k/k-" + str(i) for i in range(20, 420, 20)]
	topic = df_metadata.topic.tolist()
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

def compute_adjusted_NMI_wikitext(broad_categories=True):
	df_metadata = pd.read_json("data/wikitext/processed/labeled/vocab_15k/train.metadata.jsonl", lines=True)
	print (df_metadata)
	paths = ["runs/outputs/k_selection/" + "wikitext" + "-labeled/vocab_15k/k-" + str(i) for i in range(20, 420, 20)]

	if broad_categories:
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
	
# experiments are: ["bills_broad", "wikitext_broad", "wikitext_specific"]
 
experiment = "bills_broad"
if experiment == "bills_broad":
	dataset = "bills":
	cluster_metrics = compute_adjusted_NMI_bills()
	paths = ["runs/outputs/k_selection/bills-labeled/vocab_15k/k-" + str(i) for i in range(20, 420, 20)]
	df = pd.read_csv("LLM-scores/LLM_outputs_bills_broad.csv")
	plt_label = "adjusted NMI broad topics"
	outfile_name = "n_clusters_bills_dataset.png"
	
elif experiment == "wikitext_broad":
	dataset = "wikitext"
	cluster_metrics = compute_adjusted_NMI_wikitext()
	paths = ["runs/outputs/k_selection/wikitext-labeled/vocab_15k/k-" + str(i) for i in range(20, 420, 20)]
	df = pd.read_csv("LLM-scores/LLM_outputs_wikitext_broad.csv")
	plt_label = "adjusted NMI broad topics"
	outfile_name = "n_clusters_wikitext_broad.png"

	
elif experiment == "wikitext_specific":
	dataset = "wikitext"
	cluster_metrics = compute_adjusted_NMI_wikitext(broad_categories=False)
	paths = ["runs/outputs/k_selection/wikitext-labeled/vocab_15k/k-" + str(i) for i in range(20, 420, 20)]
	df = pd.read_csv("LLM-scores/LLM_outputs_wikitext_specific.csv")
	plt_label = "adjusted NMI specific topics"
	outfile_name = "n_clusters_wikitext_specific.png"


df["gpt_ratings"] = df.chatGPT_eval.astype(int)
average_goodness = []
# get average gpt_ratings for each k
for path in paths:
	path = os.path.join(path, "2972")
	df_at_k = df[df.path == path]
	average_goodness.append(df_at_k.gpt_ratings.mean())	

average_goodness = moving_average(average_goodness) # smooth via moving_average to remove weird outliers

# if we want to plot spearmanR for different clustering metrics
compute_spearmanR_different_cluster_metrics = False
if compute_spearmanR_different_cluster_metrics:
	for key in ["ami", "ari", "completeness", "homogeneity"]:
		value = cluster_metrics[key]
		statistic = spearmanr(average_goodness, value)
		print ("topics " + key, statistic.statistic.round(3), statistic.pvalue.round(3), )

# re-shape to compute z-scores (otherwise, the scales are off because LLM scores and clustering metrics are on different scales and the graphs do not look too great
average_goodness = np.array(average_goodness).reshape(-1, 1)
AMI = np.array(cluster_metrics["ami"]).reshape(-1, 1)

# compute z-scores
average_goodness = StandardScaler().fit_transform(average_goodness).squeeze()
AMI = StandardScaler().fit_transform(AMI).squeeze()

# plot figures
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(average_goodness, label="LLM score", color="tab:red")
plt.plot(nmis, label = plt_label, color="tab:blue")
plt.legend()
ax.set_ylabel("z-scores")
ax.set_xlabel("number of clusters")

n_clusters = list(range(20, 420, 20)) 
plt.xticks(range(len(n_clusters)), n_clusters, rotation=45)
plt.savefig(outfile_name)


