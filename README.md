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
- 