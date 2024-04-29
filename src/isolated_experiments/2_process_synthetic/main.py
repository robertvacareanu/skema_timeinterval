import json
from src.utils import parse_synthetic_data

if __name__== "__main__":
    with open('data/synthetic_original/initial_data.jsonl') as fin:
        data = json.load(fin)
    
    data_parsed = [parse_synthetic_data(x) for x in data]
    data_parsed = [x for x in data_parsed if x]

    with open('data/synthetic/initial_data.jsonl', 'w+') as fout:
        json.dump(data_parsed, fout, indent=4)
