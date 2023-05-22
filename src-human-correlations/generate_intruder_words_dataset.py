import json
import random


if __name__ == "__main__":
	with open("all_data.json") as f:
		data = json.load(f)

	random.seed(42)
	with open("intruder_outfile.jsonl", "w") as outfile:
		for dataset_name, dataset in data.items():
			for model_name, dataset_model in dataset.items():
				print (dataset_name, model_name)
				if model_name == "mallet":
					fn = "topic-modeling-output/mallet-topics-best-c_npmi_10_full.json"
				elif model_name == "dvae":
					fn = "topic-modeling-output/dvae-topics-best-c_npmi_10_full.json"
				else:
					fn = "topic-modeling-output/etm-topics-best-c_npmi_10_full.json"
				with open(fn) as f:
					topics_data = json.load(f)

				raw_topics = topics_data[dataset_name]["topics"]				

				words = set()
				for topic in raw_topics:
					words.update(topic)
				words = list(set(words))
				for i, (topic, metric, double_check) in enumerate(zip(raw_topics, dataset_model["metrics"]["intrusion_scores_avg"], dataset_model["topics"])):
					topic_set = set(topic)
					candidate_words = [i for i in words if i not in topic_set]
					random.shuffle(candidate_words)
					sampled_intruders = candidate_words[:10]
					for intruder in sampled_intruders:
						out = {}
						out["topic_id"] = i
						out["intruder_term"] = intruder
						out["topic_terms"] = topic[:10]
						out["intrusion_scores_avg"] = metric
						out["dataset_name"] = dataset_name
						out["model_name"] = model_name
						json.dump(out, outfile)
						outfile.write("\n")

