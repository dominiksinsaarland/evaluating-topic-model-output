#!/usr/bin/env python3

import matplotlib.pyplot as plt
import pandas as pd
import os
from tqdm import tqdm
import numpy as np
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, completeness_score, homogeneity_score
from scipy.stats import spearmanr
from collections import defaultdict
import fig_utils
from matplotlib.ticker import FormatStrFormatter
import argparse

args = argparse.ArgumentParser()
args.add_argument("--experiment", default="wikitext_specific")
args = args.parse_args()

os.makedirs("computed/figures", exist_ok=True)

N_CLUSTER = range(20, 420, 20)


def moving_average(a, window_size=3):
    if window_size == 0:
        return a
    out = []
    for i in range(len(a)):
        start = max(0, i - window_size)
        out.append(np.mean(a[start:i + 1]))
    return out


def compute_adjusted_NMI_bills():
    df_metadata = pd.read_json(
        "data/bills/processed/labeled/vocab_15k/train.metadata.jsonl",
        lines=True
    )
    print(df_metadata)
    paths = [
        "runs/outputs/k_selection/" + "bills" +
        "-labeled/vocab_15k/k-" + str(i)
        for i in N_CLUSTER
    ]
    topic = df_metadata.topic.tolist()
    cluster_metrics = defaultdict(list)

    for path, num_topics in tqdm(zip(paths, N_CLUSTER), total=20):
        path = os.path.join(path, "2972")
        beta = np.load(os.path.join(path, "beta.npy"))
        theta = np.load(os.path.join(path, "train.theta.npy"))
        argmax_theta = theta.argmax(axis=-1)
        cluster_metrics["ami"].append(
            adjusted_mutual_info_score(topic, argmax_theta)
        )
        cluster_metrics["ari"].append(
            adjusted_rand_score(topic, argmax_theta)
        )
        cluster_metrics["completeness"].append(
            completeness_score(topic, argmax_theta)
        )
        cluster_metrics["homogeneity"].append(
            homogeneity_score(topic, argmax_theta)
        )
    return cluster_metrics


def compute_adjusted_NMI_wikitext_old(broad_categories=True):
    df_metadata = pd.read_json(
        "data/wikitext/processed/labeled/vocab_15k/train.metadata.jsonl",
        lines=True
    )
    print(df_metadata)
    paths = [
        "runs/outputs/k_selection/" + "wikitext" +
        "-labeled/vocab_15k/k-" + str(i)
        for i in N_CLUSTER
    ]

    if broad_categories:
        topic = df_metadata.category.tolist()
    else:
        topic = df_metadata.subcategory.tolist()

    cluster_metrics = defaultdict(list)

    for path, num_topics in tqdm(zip(paths, N_CLUSTER), total=20):
        path = os.path.join(path, "2972")
        beta = np.load(os.path.join(path, "beta.npy"))
        theta = np.load(os.path.join(path, "train.theta.npy"))
        argmax_theta = theta.argmax(axis=-1)
        cluster_metrics["ami"].append(
            adjusted_mutual_info_score(topic, argmax_theta)
        )
        cluster_metrics["ari"].append(adjusted_rand_score(topic, argmax_theta))
        cluster_metrics["completeness"].append(
            completeness_score(topic, argmax_theta)
        )
        cluster_metrics["homogeneity"].append(
            homogeneity_score(topic, argmax_theta)
        )
    return cluster_metrics

def compute_adjusted_NMI_wikitext(broad_categories=True):
	df_metadata = pd.read_json("data/wikitext/processed/labeled/vocab_15k/train.metadata.jsonl", lines=True)
	print (df_metadata)
	paths = ["runs/outputs/k_selection/" + "wikitext" + "-labeled/vocab_15k/k-" + str(i) for i in range(20, 420, 20)]

	# re-do columns:
	value_counts = df_metadata.category.value_counts().rename_axis("topic").reset_index(name="counts")
	value_counts = {i:j for i,j in zip(value_counts.topic, value_counts.counts) if j > 25}
	df_metadata["subtopic"] = ["other" if i in value_counts else i for i in df_metadata.category]

	value_counts = df_metadata.subcategory.value_counts().rename_axis("topic").reset_index(name="counts")
	value_counts = {i:j for i,j in zip(value_counts.topic, value_counts.counts) if j > 25}
	df_metadata["subtopic"] = ["other" if i in value_counts else i for i in df_metadata.subcategory]

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




# experiment = "bills_broad"
# experiment = "wikitext_broad"
# experiment = "wikitext_specific"

if args.experiment == "bills_broad":
    dataset = "bills"
    cluster_metrics = compute_adjusted_NMI_bills()
    paths = [
        "runs/outputs/k_selection/bills-labeled/vocab_15k/k-" + str(i)
        for i in N_CLUSTER
    ]
    df = pd.read_csv("LLM-scores/LLM_outputs_bills_broad.csv")
    plt_label = "Adjusted MI"
    outfile_name = "n_clusters_bills_dataset.pdf"
    plt_title = "BillSum, broad, $\\rho = RHO_CORR$"
    left_ylab = True
    right_ylab = False
    degree = 1
elif args.experiment == "wikitext_broad":
    dataset = "wikitext"
    cluster_metrics = compute_adjusted_NMI_wikitext()
    paths = [
        "runs/outputs/k_selection/wikitext-labeled/vocab_15k/k-" + str(i)
        for i in N_CLUSTER
    ]
    df = pd.read_csv("LLM-scores/LLM_outputs_wikitext_broad.csv")
    plt_label = "Adjusted MI"
    outfile_name = "n_clusters_wikitext_broad.pdf"
    plt_title = "Wikitext, broad, $\\rho = RHO_CORR$"
    left_ylab = False
    right_ylab = False
    degree = 1
elif args.experiment == "wikitext_specific":
    dataset = "wikitext"
    cluster_metrics = compute_adjusted_NMI_wikitext(broad_categories=False)
    paths = [
        "runs/outputs/k_selection/wikitext-labeled/vocab_15k/k-" + str(i)
        for i in N_CLUSTER
    ]
    df = pd.read_csv("LLM-scores/LLM_outputs_wikitext_specific.csv")
    plt_label = "Adjusted MI"
    outfile_name = "n_clusters_wikitext_specific.pdf"
    plt_title = "Wikitext, specific, $\\rho = RHO_CORR$"
    left_ylab = False
    right_ylab = True
    degree = 3

df["gpt_ratings"] = df.chatGPT_eval.astype(int)
average_goodness = []
# get average gpt_ratings for each k
for path in paths:
    path = os.path.join(path, "2972")
    df_at_k = df[df.path == path]
    average_goodness.append(df_at_k.gpt_ratings.mean())

# smooth via moving_average to remove weird outliers
average_goodness = moving_average(average_goodness)

# if we want to plot spearmanR for different clustering metrics
compute_spearmanR_different_cluster_metrics = False
if compute_spearmanR_different_cluster_metrics:
    for key in ["ami", "ari", "completeness", "homogeneity"]:
        value = cluster_metrics[key]
        statistic = spearmanr(average_goodness, value)
        print(
            "topics " + key, statistic.statistic.round(3),
            statistic.pvalue.round(3)
        )

# re-shape to compute z-scores (otherwise, the scales are off because LLM scores and clustering metrics are on different scales and the graphs do not look too great
average_goodness = np.array(average_goodness).reshape(-1, 1)
AMI = np.array(cluster_metrics["ami"]).reshape(-1, 1)

# compute z-scores
# average_goodness = StandardScaler().fit_transform(average_goodness).squeeze()
# AMI = StandardScaler().fit_transform(AMI).squeeze()
average_goodness = np.array(average_goodness).squeeze()
AMI = np.array(AMI).squeeze()

# plot figures
fig = plt.figure(figsize=(3.5, 2))
ax1 = plt.gca()
ax2 = ax1.twinx()
SCATTER_STYLE = {"edgecolor": "black", "s": 30}
l1 = ax1.scatter(
    N_CLUSTER,
    average_goodness,
    label="LLM score",
    color=fig_utils.COLORS[0],
    **SCATTER_STYLE
)
l2 = ax2.scatter(
    N_CLUSTER,
    AMI,
    label=plt_label,
    color=fig_utils.COLORS[1],
    **SCATTER_STYLE
)

# print to one digit
ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

xticks_fine = np.linspace(min(N_CLUSTER), max(N_CLUSTER), 500)

poly_ag = np.poly1d(np.polyfit(N_CLUSTER, average_goodness, degree))
ax1.plot(xticks_fine, poly_ag(xticks_fine), '-', color=fig_utils.COLORS[0], zorder=-100)
poly_ami = np.poly1d(np.polyfit(N_CLUSTER, AMI, degree))
ax2.plot(xticks_fine, poly_ami(xticks_fine), '-', color=fig_utils.COLORS[1], zorder=-100)

plt.legend(
    [l1, l2], [p_.get_label() for p_ in [l1, l2]],
    loc="upper right",
    handletextpad=-0.2,
    labelspacing=0.15,
    borderpad=0.15,
    borderaxespad=0.1,
)
if left_ylab:
    ax1.set_ylabel("Adjusted MI")
if right_ylab:
    ax2.set_ylabel("Averaged LLM score")
ax1.set_xlabel("Number of topics")
plt.xticks(N_CLUSTER[::3], N_CLUSTER[::3])


statistic = spearmanr(average_goodness, cluster_metrics["ami"])
plt.title(plt_title.replace("RHO_CORR", f"{statistic[0]:.2f}"))

plt.tight_layout(pad=0.2)
plt.savefig("computed/figures/" + outfile_name)
plt.show()
