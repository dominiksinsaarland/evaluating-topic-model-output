#!/usr/bin/env python3

import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import numpy as np
from collections import defaultdict
from scipy.stats import spearmanr
import fig_utils
from matplotlib.ticker import FormatStrFormatter
import csv
import argparse

args = argparse.ArgumentParser()
args.add_argument("--experiment", default="wikitext_specific")
args = args.parse_args()

os.makedirs("computed/figures", exist_ok=True)

N_CLUSTER = range(20, 420, 20)

data = list(csv.DictReader(
    open(f"LLM-scores-3/{args.experiment}_dataframe_all_results.csv", "r")))

if args.experiment == "bills_broad":
    dataset = "bills"
    outfile_name = "n_clusters_bills_broad.pdf"
    plt_title = "BillSum, broad, $\\rho_D = RHO_CORR1$ $\\rho_W = RHO_CORR2$"
    left_ylab = True
    show_legend = True
    degree = 4
elif args.experiment == "wikitext_broad":
    dataset = "wikitext"
    outfile_name = "n_clusters_wikitext_broad.pdf"
    plt_title = "Wikitext, broad, $\\rho_D = RHO_CORR1$ $\\rho_W = RHO_CORR2$"
    left_ylab = False
    show_legend = False
    degree = 4
elif args.experiment == "wikitext_specific":
    dataset = "wikitext"
    outfile_name = "n_clusters_wikitext_specific.pdf"
    plt_title = "Wikitext, specific, $\\rho_D = RHO_CORR1$ $\\rho_W = RHO_CORR2$"
    left_ylab = False
    show_legend = False
    degree = 4

data_llm_word = [float(x["LLM Scores Wordset"]) for x in data]
data_llm_doc = [float(x["LLM Scores Documentset"]) for x in data]
data_ari = [float(x["ARI"]) for x in data]

# plot figures
fig = plt.figure(figsize=(3.5, 2))
ax1 = plt.gca()
ax2 = ax1.twinx()
ax3 = ax1.twinx()
SCATTER_STYLE = {"edgecolor": "black", "s": 30}
l1 = ax1.scatter(
    N_CLUSTER,
    data_ari,
    label="ARI",
    color=fig_utils.COLORS[1],
    **SCATTER_STYLE
)
l2 = ax2.scatter(
    N_CLUSTER,
    data_llm_word,
    label="Word LLM",
    color=fig_utils.COLORS[0],
    **SCATTER_STYLE
)
l3 = ax3.scatter(
    N_CLUSTER,
    data_llm_doc,
    label="Doc LLM",
    color=fig_utils.COLORS[4],
    **SCATTER_STYLE
)

# ax1.axes.get_yaxis().set_visible(False)
# print to one digit
# ax1.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
ax1.set_yticks([])
ax2.set_axis_off()
ax3.set_axis_off()

xticks_fine = np.linspace(min(N_CLUSTER), max(N_CLUSTER), 500)

poly_ami = np.poly1d(np.polyfit(N_CLUSTER, data_ari, degree))
ax1.plot(
    xticks_fine, poly_ami(xticks_fine), '-', color=fig_utils.COLORS[1], zorder=-100
)
poly_llm_word = np.poly1d(np.polyfit(N_CLUSTER, data_llm_word, degree))
ax2.plot(
    xticks_fine, poly_llm_word(xticks_fine), '-', color=fig_utils.COLORS[0], zorder=-100
)
poly_llm_doc = np.poly1d(np.polyfit(N_CLUSTER, data_llm_doc, degree))
ax3.plot(
    xticks_fine, poly_llm_doc(xticks_fine), '-', color=fig_utils.COLORS[4], zorder=-100
)

if show_legend:
    lhandles = [l2, l3, l1]
    plt.legend(
        lhandles, [p_.get_label() for p_ in lhandles],
        loc="upper right",
        handletextpad=-0.2,
        labelspacing=0.1,
        borderpad=0.2,
        borderaxespad=0.2,
        handlelength=1.5,
        columnspacing=0.8,
        ncols=2,
        edgecolor="black",
        facecolor="#dddddd"
    )
if left_ylab:
    ax1.set_ylabel("Metric Scores")
else:
    ax1.set_ylabel(" ")

ax1.set_xlabel("Number of topics")
plt.xticks(N_CLUSTER[::3], N_CLUSTER[::3])


statistic_doc = spearmanr(data_llm_doc, data_ari)
statistic_word = spearmanr(data_llm_word, data_ari)
# statistic = spearmanr(data_llm_doc, data_llm_word)
plt.title(
    plt_title
    .replace("RHO_CORR1", f"{statistic_doc[0]:.2f}")
    .replace("RHO_CORR2", f"{statistic_word[0]:.2f}")
)

plt.tight_layout(pad=0.0)
plt.savefig("computed/figures/" + outfile_name)
plt.show()
