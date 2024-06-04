import argparse

def get_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1, help="Random Seed")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight Decay")
    parser.add_argument("--model_name", type=str, default="t5-base", help="Model name")
    parser.add_argument("--saving_path", type=str, help="Where to save logs, in the form of a `.jsonl` file. Also uses this name to save the model during training `output/{saving_path}`")
    parser.add_argument("--training_steps", type=int, default=10000, help="The number of training steps (gradient updates)")
    parser.add_argument("--learning_rate", type=float, default=3e-5, help="The learning rate")
    parser.add_argument("--use_original", action='store_true', help="If set, we will use original data for training. We evaluate on original data regardless of the status of this flag")
    parser.add_argument("--use_curated", action='store_true', help="If set, we will use curated data for training. We evaluate on original data regardless of the status of this flag")
    parser.add_argument("--use_paraphrase", action='store_true', help="If set, use paraphrase data for training")
    parser.add_argument("--use_synthetic", action='store_true', help="If set, use synthetic data for training")
    parser.add_argument("--print_debug", action='store_true', )
    parser.add_argument("--debug_saving_path", type=str, help="Where to save the debug outputs")

    parser.add_argument("--per_device_train_batch_size", type=int, default=4, help="Batch size for training (default=4)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient Accumulation")

    return parser

