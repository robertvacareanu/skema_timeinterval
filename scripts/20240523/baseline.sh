#!/bin/bash


#### NOTES ####

# (1) All the ideas that have been floating around
# - Training T5 on the vanilla annotations
# - Rephrasing stuff
# - Adding synthetic data
# - Others models we've tried (BART, Larger T5, Flan T5)
# - Describe all the steps to fix the annotations: (i) normalize dates, (ii) locations that mention `city, state`
# - How did we evaluate
# - Joint VS Independently

# (2) Table 
# Method | Location F1 | Time F1
# ------------------------------
# Original | |
# Original + Paraphrase | |
# Original + Synthetic | |

#### NOTES ####


index=1
path=results/baseline-original/
mkdir -p $path
for seed in 1 2 3
do
    python -m src.all_srl_baseline --seed $seed  --use_original >> ${path}/baseline_results_${seed}.json
    index=$((index + 1))
done





# # (2) Train T5 on vanilla + paraphrase
# index=1
# path=results/240523/original_paraphrase_specific
# mkdir -p $path
# for seed in 1 2 3
# do
#     for steps in 1000 2000 5000 10000
#     do
#         CUDA_VISIBLE_DEVICES=0 python -m src.t5_specific_event --seed $seed --weight_decay 0.1 --model_name t5-base --saving_path ${path}/results_${index} --training_steps $steps --use_original --use_paraphrase >> ${path}/results_${index}.txt
#         index=$((index + 1))
#     done
# done

# index=1
# path=results/240523/original_paraphrase_all
# mkdir -p $path
# for seed in 1 2 3
# do
#     for steps in 1000 2000 5000 10000
#     do
#         CUDA_VISIBLE_DEVICES=0 python -m src.t5_all_events --seed $seed --weight_decay 0.1 --model_name t5-base --saving_path ${path}/results_${index} --training_steps $steps --use_original --use_paraphrase >> ${path}/results_${index}.txt
#         index=$((index + 1))
#     done
# done




