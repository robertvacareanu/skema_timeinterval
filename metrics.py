from collections import defaultdict
import numpy as np

def evaluate_individual_sets(gold_standard, predicted):
    all_keys = gold_standard.keys()
    result = {}
    for key in all_keys:
        if key not in predicted:
            result[key] = {
                'precision': 0.0, 
                'recall'   : 0.0,
                'f1_score' : 0.0, 
            }
            continue
        
        gold_standard_set = set([x.lower() for x in gold_standard[key]])
        predicted_set = set([x.lower() for x in predicted[key]])
        
        # Calculate intersection, union, and difference of sets
        intersection = gold_standard_set.intersection(predicted_set)
        
        # Calculate precision, recall, and F1-score
        precision = len(intersection) / len(predicted_set) if len(predicted_set) > 0 else 0
        recall = len(intersection) / len(gold_standard_set) if len(gold_standard_set) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        result[key] = {
            'precision': precision, 
            'recall'   : recall,
            'f1_score' : f1_score, 
        }

    return result


def evaluate_sets(gold, pred):
    results  = [evaluate_individual_sets(g, p) for (g, p) in zip(gold, pred)]

    scores = defaultdict(list)
    for x in results:
        for (key, value) in x.items():
            if key not in scores:
                scores[key] = {
                    'precision': [],
                    'recall'   : [],
                    'f1_score' : [],
                }
            scores[key]['precision'].append(value['precision'])
            scores[key]['recall'].append(value['recall'])
            scores[key]['f1_score'].append(value['f1_score'])


    return [{'key': key, 'precision': np.mean(values['precision']), 'recall': np.mean(values['recall']), 'f1_score': np.mean(values['f1_score'])} for (key, values) in scores.items()]
    



if __name__ == "__main__":
    gold_standard = {
        'location': ['Wuhan'],
        'time': ['january 21', 'february 4, 2020', 'march 20']
    }

    predicted = {
        'location': ['Wuhan'],
        'time': ['february 4, 2021', 'january 21', 'march 20']
    }

    result = evaluate_individual_sets(gold_standard, predicted)
    from pprint import pprint
    pprint(result, width=120)
