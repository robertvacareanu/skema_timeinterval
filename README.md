# skema_timeinterval

This project contains the necessary code to train a T5 model to extract specific details (hereafter called contents) related to a given input text (hereafter called text).
The contents we are interested in are "location" and "time". 
There are two approaches investigated in this repository:

(1) Extracting (or generating) the full contents in one go, token by token.

(2) Generating the text for a specific type of content (e.g., "location", "time"). The full contents are generated, then, by generating the output individually, then assembling.


### Running Commands

```
CUDA_VISIBLE_DEVICES=0 python -i -m src.t5_specific_event --seed 1 --weight_decay 0.1 --model_name t5-base --saving_path results/240517/results_paraphrase_synthetic_240517 --use_paraphrase --use_synthetic --training_steps 1000  --use_original >> results/240517/results_paraphrase_synthetic_240517.txt
```

#### Flags
- `seed` -> The random seed to use; Used for: (1) Setting `transformers` seed (with `transformers.set_seed()`) and (2) Random object, for shuffling and data splitting
- `weight_decay` -> The weight decay parameter, used with Huggingface transformers `Trainer`
- `model_name` -> The name of the model, to be loaded (with `AutoModelForSeq2SeqLM.from_pretrained(model_name)`)
- `saving_path` -> Where to save the results. Used in (1) `Seq2SeqTrainingArguments` (`outputs/{saving_path}`), (2) Saving debug lines
- `training_steps` -> For how many steps to train for
- `learning_rate` -> The learning rate
- `use_original` -> Whether to use the original data for training; It is used for testing regardless of the status of this flag
- `use_paraphrase` -> Whether to use the paraphrase data for training and testing
- `use_synthetic` -> Whether to use the synthetic data for training and testing