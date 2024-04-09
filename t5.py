from transformers import AutoTokenizer

import glob
import json
import random
import pandas as pd

from sklearn.model_selection import train_test_split
import datasets
from datasets import load_dataset
from transformers import DataCollatorForSeq2Seq

from datetime import datetime
from metrics import evaluate_sets
from make_data_structured import line_to_dict
from src.utils import get_hash, remove_unnecessary_content

# for weight_decay in [0.1, 0.01, 0.001]:
    # for model_name in ['t5-small', 't5-base']:
        # for number_of_epochs in [10, 20, 50, 100, 200, 500, 1000, 2000]:
for seed in [1, 2, 3, 4, 5]:
    for weight_decay in [0.1]:
        for model_name in ['t5-base']:
            for number_of_epochs in [25]:
                r = random.Random(seed)
                data = []
                for f in glob.glob('data/*.json'):
                    with open(f) as fin:
                        # loaded = json.load(fin)
                        # for l in loaded:
                            # data += {**loaded, 'original': get_hash(loaded)}
                        data += json.load(fin)
                
                for f in glob.glob('data_paraphrased_*.jsonl'):
                    with open(f) as fin:
                        for line in fin:
                            data.append(json.loads(line))

                data = [x for x in data if all(y in x.keys() for y in ['contents', 'text', 'pre_context', 'post_context'])]
                banned_contents_words = ['AUTHOR_INST', 'DATE', 'STRENGTHS', 'DESCRIPTION', 'USAGE', 'ASSUMPTIONS', 'All these instances are very', 'is relative temporal', 'AUTHOR_AUTHOR', 'AUTHOR', 'DATASET', 'DATASET', 'SCHEMA', "variation"]
                for w in banned_contents_words:
                    data = [x for x in data if w not in x['contents']]
                
                data = [{**x, 'original': get_hash(x)} if 'original' not in x else x for x in data]
                data = [remove_unnecessary_content(x) for x in data]

                all_hashes = sorted(list(set([x['original'] for x in data])))
                all_hashes = r.sample(all_hashes, k=len(all_hashes))
                train_hashes = set(all_hashes[:int(len(all_hashes) * 0.8)])
                test_hashes  = set(all_hashes[len(train_hashes):])

                train = [x for x in data if x['original'] in train_hashes]
                test  = [x for x in data if x['original'] in test_hashes]

                train = [{
                    'input': 'Text:\n' + x['text'] + '\n\nContext:\n' + x['pre_context'] + '\n' + x['post_context'],
                    'output': x['contents']
                } for x in train]
                test = [{
                    'input': 'Text:\n' + x['text'] + '\n\nContext:\n' + x['pre_context'] + '\n' + x['post_context'],
                    'output': x['contents']
                } for x in test]

                data = datasets.DatasetDict({
                    'train': datasets.Dataset.from_list(train),
                    'test' : datasets.Dataset.from_list(test),
                })

                def preprocess_function(examples):

                    # Tokenizes the prepended input texts to convert them into a format that can be fed into the T5 model.
                    # Sets a maximum token length of 1024, and truncates any text longer than this limit.
                    model_inputs = tokenizer(examples["input"], max_length=1024, truncation=True)

                    labels = tokenizer(text_target=examples["output"], max_length=128, truncation=True)

                    # Assigns the tokenized labels to the 'labels' field of model_inputs.
                    # The 'labels' field is used during training to calculate the loss and guide model learning.
                    model_inputs["labels"] = labels["input_ids"]

                    # Returns the prepared inputs and labels as a single dictionary, ready for training.
                    return model_inputs
                    

                tokenizer = AutoTokenizer.from_pretrained(model_name)

                tokenized = data.map(preprocess_function, batched=True)



                from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer

                data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model_name)
                model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


                training_args = Seq2SeqTrainingArguments(
                    output_dir="outputs/240329",
                    evaluation_strategy='no',
                    learning_rate=1e-5,
                    per_device_train_batch_size=4,
                    per_device_eval_batch_size=4,
                    weight_decay=weight_decay,
                    save_total_limit=1,
                    save_strategy = "no",
                    num_train_epochs=number_of_epochs,
                    # fp16=True,
                )

                trainer = Seq2SeqTrainer(
                    model=model,
                    args=training_args,
                    train_dataset=tokenized["train"],
                    tokenizer=tokenizer,
                    data_collator=data_collator,
                )

                trainer.train()
                trainer.save_model('outputs/240329')

                model = AutoModelForSeq2SeqLM.from_pretrained("outputs/240329")

                import torch

                train_expected    = []
                train_generations = []
                for train in tokenized['train']:
                    train_generations.append(tokenizer.decode(model.generate(input_ids=torch.tensor([train['input_ids']]).to(model.device)).tolist()[0], skip_special_tokens=True))
                    train_expected.append(train['output'])

                train_expected    = [line_to_dict(x) for x in train_expected]
                train_generations = [line_to_dict(x) for x in train_generations]

                    
                test_expected    = []
                test_generations = []
                for test in tokenized['test']:
                    test_generations.append(tokenizer.decode(model.generate(input_ids=torch.tensor([test['input_ids']]).to(model.device)).tolist()[0], skip_special_tokens=True))
                    test_expected.append(test['output'])

                test_expected    = [line_to_dict(x) for x in test_expected]
                test_generations = [line_to_dict(x) for x in test_generations]

                train_results = evaluate_sets(train_expected, train_generations)
                test_results  = evaluate_sets(test_expected, test_generations)
                with open('results_2403029.jsonl', 'a+') as fout:
                    _=fout.write(json.dumps({
                        'time'            : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'seed'            : seed,
                        'weight_decay'    : weight_decay,
                        'model_name'      : model_name,
                        'number_of_epochs': number_of_epochs,
                        'train_results'   : train_results, 
                        'test'            : test_results
                    }))
                    _=fout.write('\n')


