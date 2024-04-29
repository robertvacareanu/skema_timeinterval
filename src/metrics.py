from collections import defaultdict
import numpy as np

def evaluate_individual_sets(gold_standard, predicted, replace_chars = True):
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
        
        gold_standard_processed = [x.lower().strip() for x in gold_standard[key]]
        predicted_processed     = [x.lower().strip() for x in predicted[key]]

        if replace_chars:
            gold_standard_processed = [x.lower().replace(',', '').replace('.', '') for x in gold_standard_processed]
            predicted_processed     = [x.lower().replace(',', '').replace('.', '') for x in predicted_processed]

        gold_standard_set = set(gold_standard_processed)
        predicted_set     = set(predicted_processed)
        
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


def evaluate_individual_sets_per_token(gold_standard, predicted, replace_chars = True):
    """
    Score between two dictionaries, containing keys (usually time and location).
    And we have that gold_standard[key] and predicted[key] are lists of strings.
    """
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
        
        gold_standard_processed = [x.lower().strip() for x in gold_standard[key]]
        predicted_processed     = [x.lower().strip() for x in predicted[key]]

        if replace_chars:
            gold_standard_processed = [x.lower().replace(',', '').replace('.', '') for x in gold_standard_processed]
            predicted_processed     = [x.lower().replace(',', '').replace('.', '') for x in predicted_processed]

        gold_standard_processed = [y for x in gold_standard_processed for y in x.split()]
        predicted_processed = [y for x in predicted_processed for y in x.split()]

        precision, recall, f1_score = token_f1_score(gold_standard_processed, predicted_processed)
        
        
        result[key] = {
            'precision': precision, 
            'recall'   : recall,
            'f1_score' : f1_score, 
        }

    return result




def evaluate_sets(gold, pred, evaluate_at_token_level: bool = False):
    results  = [evaluate_individual_sets_per_token(g, p) if evaluate_at_token_level else evaluate_individual_sets(g, p) for (g, p) in zip(gold, pred)]

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
    

def token_f1_score(list1, list2):
    """
    Token-level F1 score
    """
    set1 = set(list1)
    set2 = set(list2)

    # True positives are the intersection of the two sets
    tp = len(set1 & set2)

    # False positives are items in set1 but not in set2
    fp = len(set1 - set2)

    # False negatives are items in set2 but not in set1
    fn = len(set2 - set1)

    # Precision is the ratio of true positives to the sum of true and false positives
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0

    # Recall is the ratio of true positives to the sum of true positives and false negatives
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    # F1 score is the harmonic mean of precision and recall
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1



if __name__ == "__main__":
    gold_standard = {
        'location': ['Wuhan'],
        'time': ['january 21', 'february 4, 2020', 'march 20']
    }

    predicted = {
        'location': ['Wuhan'],
        'time': ['february 4, 2021', 'january 21', 'march 20']
    }

    from pprint import pprint
    pprint(evaluate_individual_sets(gold_standard, predicted), width=120)
    pprint(evaluate_individual_sets_per_token(gold_standard, predicted), width=120)
