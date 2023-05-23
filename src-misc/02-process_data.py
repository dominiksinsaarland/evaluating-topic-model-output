#!/usr/bin/env python3

import json

data = json.load(open("data/intrusion.json", "r"))

for topic_scores, topic_words in zip(data["wikitext"]["etm"]["metrics"]["intrusion_scores_raw"], data["wikitext"]["etm"]["topics"]):
    print(len(topic_scores), len(topic_words))
    group_a = [w for s, w in zip(topic_scores, topic_words) if s == 0]
    group_b = [w for s, w in zip(topic_scores, topic_words) if s == 1]
    print(group_a, group_b)