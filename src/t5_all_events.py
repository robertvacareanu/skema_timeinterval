"""
A regulat seq2seq app, aiming to:
    (1) Take "input", which is a concatenation of text and context
    (2) Generate "output", which represents the contents

The model can be trained on a mix of original, paraphrased, and synthetic data
"""
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, set_seed

import glob
import json
import tqdm
import random

import torch
import datasets

from datetime import datetime
from src.metrics import evaluate_sets
from src.make_data_structured import line_to_dict
from src.utils import get_hash, remove_unnecessary_content, preprocess_function
from src.metrics import evaluate_individual_sets_per_token
from src.parser_utils import get_parser

import wandb

wandb.init(mode='disabled')


source_id2name = {
    0: 'original',
    1: 'paraphrase',
    2: 'synthetic',
}

source_name2id = {v:k for (k, v) in source_id2name.items()}

parser = get_parser()

args = vars(parser.parse_args())

seed           = args['seed']
weight_decay   = args['weight_decay']
model_name     = args['model_name']
training_steps = args['training_steps']
learning_rate  = args['learning_rate']
saving_path    = args['saving_path']

set_seed(seed)
r = random.Random(seed)
original_data = []

##########################
### START READING DATA ###
##########################

# Read original
for f in glob.glob('data/original/*.json'):
    with open(f) as fin:
        original_data += json.load(fin)
        original_data = [{**x, 'source': source_name2id['original']} for x in original_data]

# Read paraphrases (if needed)
if args['use_paraphrase']:
    for f in glob.glob('data/paraphrases/240423/*.jsonl'):
        with open(f) as fin:
            for line in fin:
                original_data.append({**json.loads(line), 'source': source_name2id['paraphrase']})

# Read synthetic (if needed)
if args['use_synthetic']:
    for f in glob.glob('data/synthetic/*.jsonl'):
        with open(f) as fin:
            loaded = json.load(fin)
            original_data += [{**x, 'source': source_name2id['synthetic']} for x in loaded]

########################
### END READING DATA ###
########################



####################
### START FILTER ###
####################


# Keep only the data that contains all the fields: ['contents', 'text', 'pre_context', 'post_context']
original_data = [x for x in original_data if all(y in x.keys() for y in ['contents', 'text', 'pre_context', 'post_context'])]
# Skip some of the data
banned_contents_words = ['AUTHOR_INST', 'DATE', 'STRENGTHS', 'DESCRIPTION', 'USAGE', 'ASSUMPTIONS', 'All these instances are very', 'is relative temporal', 'AUTHOR_AUTHOR', 'AUTHOR', 'DATASET', 'DATASET', 'SCHEMA', "variation"]
for w in banned_contents_words:
    original_data = [x for x in original_data if w not in x['contents']]

##################
### END FILTER ###
##################



###########################
### START PREPROCESSING ###
###########################

# Set the "text" as the "original" field, which will be used as some form of "hash"
original_data = [{**x, 'original': x['text']} for x in original_data]
original_data = [remove_unnecessary_content(x) for x in original_data]
original_data = [x for x in original_data if len(x['contents'].split(" ")) < 8]

# Get all hashes, then shuffle
all_hashes = sorted(list(set([x['original'] for x in original_data])))
all_hashes = r.sample(all_hashes, k=len(all_hashes))

# The hashes of human annotated data
original_hashes = [x['original'] for x in original_data if x['source'] == source_name2id['original']]

if args['use_original'] is False:
    all_hashes = [x for x in all_hashes if x not in original_hashes]
    train_hashes = set(all_hashes[:int(len(all_hashes) * 0.8)])
    test_hashes  = set(all_hashes[len(train_hashes):] + original_hashes)
else:
    train_hashes = set(all_hashes[:int(len(all_hashes) * 0.8)])
    test_hashes  = set(all_hashes[len(train_hashes):])

train = [x for x in original_data if x['original'] in train_hashes]
test  = [x for x in original_data if x['original'] in test_hashes]

if args['use_original'] is False:
    train = [x for x in original_data if x['source'] != source_name2id['original']]

train = [{
    'input': 'Text:\n' + x['text'] + '\n\nContext:\n' + x['pre_context'] + x['text'] + x['post_context'],
    'output': x['contents'],
    'source': x['source']
} for x in train]
test = [{
    'input': 'Text:\n' + x['text'] + '\n\nContext:\n' + x['pre_context'] + x['text'] + x['post_context'],
    'output': x['contents'],
    'source': x['source']
} for x in test]

data = datasets.DatasetDict({
    'train': datasets.Dataset.from_list(train),
    'test' : datasets.Dataset.from_list(test),
})

tokenizer = AutoTokenizer.from_pretrained(model_name)

tokenized = data.map(lambda examples: preprocess_function(examples, tokenizer), batched=True)

print(tokenized)

#########################
### END PREPROCESSING ###
#########################






######################
### START TRAINING ###
######################

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


training_args = Seq2SeqTrainingArguments(
    output_dir=f"outputs/{saving_path}",
    evaluation_strategy='no',
    learning_rate=learning_rate,
    per_device_train_batch_size=args['per_device_train_batch_size'],
    gradient_accumulation_steps=args['gradient_accumulation_steps'],
    per_device_eval_batch_size=4,
    weight_decay=weight_decay,
    save_total_limit=1,
    save_strategy = "no",
    max_steps=training_steps,
    # num_train_epochs=number_of_epochs,
    # fp16=True,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"].remove_columns(['source']),
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()
trainer.save_model(f'outputs/{saving_path}')

####################
### END TRAINING ###
####################


model = AutoModelForSeq2SeqLM.from_pretrained(f"outputs/{saving_path}").cuda()

# train_expected    = []
# train_generations = []
# # max_length = []
# for batch in tqdm.tqdm(torch.utils.data.DataLoader(tokenized['train'].remove_columns(['input', 'output']), batch_size=32, collate_fn = data_collator)):
#     generations = model.generate(input_ids=batch['input_ids'].to(model.device), max_length=30)
#     decoded     = tokenizer.batch_decode(generations, skip_special_tokens=True)
#     gold        = tokenizer.batch_decode([[y for y in x if y != -100] for x in batch['labels'].tolist()], skip_special_tokens=True)
#     # max_length += [[y for y in x if y != -100] for x in batch['labels'].tolist()]
#     train_expected    += decoded
#     train_generations += gold


# train_expected    = [line_to_dict(x) for x in train_expected]
# train_generations = [line_to_dict(x) for x in train_generations]

    
# test = datasets.Dataset.from_list(sorted(tokenized['test'].to_list(), key=lambda x: len(x['input_ids']))).remove_columns(['input', 'output'])

test_input_ids   = []
test_expected    = []
test_generations = []
test_source      = []

for batch in tqdm.tqdm(torch.utils.data.DataLoader(tokenized['test'].remove_columns(['input', 'output']), batch_size=32, collate_fn = data_collator)):
    generations = model.generate(input_ids=batch['input_ids'].to(model.device), max_length=30)
    decoded     = tokenizer.batch_decode(generations, skip_special_tokens=True)
    gold        = tokenizer.batch_decode([[y for y in x if y != -100] for x in batch['labels'].tolist()], skip_special_tokens=True)
    test_generations += decoded
    test_input_ids   += tokenizer.batch_decode([[y for y in x if y != -100] for x in batch['input_ids'].tolist()], skip_special_tokens=True)
    test_expected    += gold
    test_source      += [source_id2name[x] for x in batch['source'].detach().cpu().tolist()]

from langchain_community.callbacks import get_openai_callback
# Transform the generated text into a structured format
with get_openai_callback() as cb1:
    test_expected    = [line_to_dict(x, should_normalize_date=True) for x in test_expected]
    test_generations = [line_to_dict(x, should_normalize_date=True) for x in test_generations]

print(cb1)

train_results = {}#evaluate_sets(train_expected, train_generations)
test_results_original   = evaluate_sets([x for (x, source) in zip(test_expected, test_source) if source == 'original'], [x for (x, source) in zip(test_generations, test_source) if source == 'original'])
test_results_synthetic  = evaluate_sets([x for (x, source) in zip(test_expected, test_source) if source == 'synthetic'], [x for (x, source) in zip(test_generations, test_source) if source == 'synthetic'])
test_results_paraphrase = evaluate_sets([x for (x, source) in zip(test_expected, test_source) if source == 'paraphrase'], [x for (x, source) in zip(test_generations, test_source) if source == 'paraphrase'])
test_results_overall    = evaluate_sets(test_expected, test_generations)

test_results_original_token_level   = evaluate_sets([x for (x, source) in zip(test_expected, test_source) if source == 'original'], [x for (x, source) in zip(test_generations, test_source) if source == 'original'], evaluate_at_token_level=True)
test_results_synthetic_token_level  = evaluate_sets([x for (x, source) in zip(test_expected, test_source) if source == 'synthetic'], [x for (x, source) in zip(test_generations, test_source) if source == 'synthetic'], evaluate_at_token_level=True)
test_results_paraphrase_token_level = evaluate_sets([x for (x, source) in zip(test_expected, test_source) if source == 'paraphrase'], [x for (x, source) in zip(test_generations, test_source) if source == 'paraphrase'], evaluate_at_token_level=True)
test_results_overall_token_level    = evaluate_sets(test_expected, test_generations, evaluate_at_token_level=True)

with open(f'{saving_path}.jsonl', 'a+') as fout:
    _=fout.write(json.dumps({
        'time'            : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'args'            : args,

        'test_original'   : sorted(test_results_original, key=lambda x: x['key']),
        'test_synthetic'  : sorted(test_results_synthetic, key=lambda x: x['key']),
        'test_paraphrase' : sorted(test_results_paraphrase, key=lambda x: x['key']),
        'test_overall'    : sorted(test_results_overall, key=lambda x: x['key']),
        
        'test_original_token_level'   : sorted(test_results_original_token_level, key=lambda x: x['key']),
        'test_synthetic_token_level'  : sorted(test_results_synthetic_token_level, key=lambda x: x['key']),
        'test_paraphrase_token_level' : sorted(test_results_paraphrase_token_level, key=lambda x: x['key']),
        'test_overall_token_level'    : sorted(test_results_overall_token_level, key=lambda x: x['key']),
    }))
    _=fout.write('\n')


if args['print_debug']:
    with open(f'{args["debug_saving_path"]}.json', 'a+') as fout:
        _=fout.write(json.dumps({
            'time'            : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'args'            : args,
            'predictions'     : [{'input_ids': tii, 'expected': te, 'generation': tg, 'score': evaluate_individual_sets_per_token(te, tg), 'source': ts} for tii, te, tg, ts in zip(test_input_ids, test_expected, test_generations, test_source)],
        }, indent=4))
        _=fout.write('\n')

    time_total            = 0
    location_total        = 0
    time_not_in_total     = 0
    location_not_in_total = 0
    for tg, tii in zip(test_generations, test_input_ids):
        for loc in tg.get('location', []):
            location_total += 1
            if loc.lower() not in tii.lower():
                location_not_in_total += 1
        for time in tg.get('time', []):
            time_total += 1
            if time.lower() not in tii.lower():
                time_not_in_total += 1
                        

    print("Total number of time entries    :", time_total)
    print("Total number of location entries:", location_total)
    print("Total number of time entries not in original text    :", time_not_in_total, ";", time_not_in_total/time_total)
    print("Total number of location entries not in original text:", location_not_in_total, ";", location_not_in_total/location_total)


    all_times     = []
    all_locations = []
    for tg, tii in zip(test_generations, test_input_ids):
        for loc in tg.get('location', []):
            all_locations.append(loc)
        for time in tg.get('time', []):
            all_times.append(time)
                        

# with open('results/240416/output.txt', 'w+') as fout:
#     for tgen, tgold, tinpt in zip(test_generations, test_expected, test_input_ids):
#         _=fout.write('-'*10)
#         _=fout.write("\n")
#         _=fout.write("Generated:")
#         _=fout.write("\n")
#         _=fout.write(json.dumps(tgen))
#         _=fout.write("\n")
#         _=fout.write("Expected :")
#         _=fout.write("\n")
#         _=fout.write(json.dumps(tgold))
#         _=fout.write("\n")
#         _=fout.write("Text")
#         _=fout.write("\n")
#         _=fout.write(tinpt)
#         _=fout.write("\n")
#         _=fout.write("Score")
#         _=fout.write("\n")
#         _=fout.write(json.dumps(evaluate_individual_sets_per_token(tgold, tgen)))
#         _=fout.write('-'*10)
#         _=fout.write("\n")