
def count(data):
	# Compute TP, FP, TN, FN for LOC and TMP

	ret = {
		"loc": {
			"tp":0,
			"tn":0,
			"fp":0,
			"fn":0,
		},
		"tmp": {
			"tp":0,
			"tn":0,
			"fp":0,
			"fn":0,
		}
	}

	preds = data.get("preds", [])
	# First, LOC
	
	# If gold is missing and pred is missing, TN
	if len(data["locations"]) == 0 and "ARGM-LOC" not in preds:
		ret['loc']["tn"] += 1
	# If gold is missing and there is a pred, FP
	elif len(data["locations"]) == 0 and "ARGM-LOC" in preds:
		ret['loc']["fp"] += 1
	# If gold exists and pred is missing, FN
	elif len(data["locations"]) > 0 and "ARGM-LOC" not in preds:
		ret['loc']["fn"] += 1
	else:
		match = False
		
		# To check for agreement, if any of the  references match lets consider it a hit, this is lenient
		pred = preds["ARGM-LOC"].lower().strip().replace(" ,", "")
		for gt in data['locations']:
			gt = gt.lower().strip()

			match = gt in pred or pred in gt
			if match:
				break


		# If gold exists and pred is wrong, FP
		if not match:
			ret['loc']["fp"] += 1
		# If gold exists and pred agree, TP
		else:
			ret['loc']["tp"] += 1

	# Second, TMP
	
	# If gold is missing and pred is missing, TN
	if len(data["temporals"]) == 0 and "ARGM-TMP" not in preds:
		ret['tmp']["tn"] += 1
	# If gold is missing and there is a pred, FP
	elif len(data["temporals"]) == 0 and "ARGM-TMP" in preds:
		ret['tmp']["fp"] += 1
	# If gold exists and pred is missing, FN
	elif len(data["temporals"]) > 0 and "ARGM-TMP" not in preds:
		ret['tmp']["fn"] += 1
	else:
		match = False
		
		# To check for agreement, if any of the  references match lets consider it a hit, this is lenient
		pred = preds["ARGM-TMP"].lower().strip().replace(" ,", "")
		for gt in data['temporals']:
			gt = gt.lower().strip()

			match = gt in pred or pred in gt
			if match:
				break


		# If gold exists and pred is wrong, FP
		if not match:
			ret['tmp']["fp"] += 1
		# If gold exists and pred agree, TP
		else:
			ret['tmp']["tp"] += 1

	return ret


def accumulate(counts):
	c = counts[0]
	c["overall"] = {
		"tp":0,
		"fp":0,
		"tn":0,
		"fn":0
	}
	for d in counts[1:]:
		for type_ in d:
			for k, v in d[type_].items():
				c[type_][k] += v
				c["overall"][k] += v

	return c

def metrics(d):
	precision = d["tp"]/ (d["tp"] + d["fp"])
	recall = d["tp"]/( d["tp"] + d["fn"])
	f1 = (precision*recall)/(precision + recall)

	return {"p": precision, "r": recall, "f1":f1}


def compute(data):
	counts = accumulate([count(d) for d in data])
	ret = {}
	for type_, vals in counts.items():
		ret[type_] = metrics(vals)

	return ret



import json

with open("test.json") as f:
	data = json.load(f)

print(compute(data))

