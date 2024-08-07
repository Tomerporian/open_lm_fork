"""Script to evaluate a model on a validation set. Based on scripts/generate.py from open_lm repo.
"""
import argparse
import json
import re

import torch

from open_lm.evaluate import evaluate_loop
from open_lm.data import get_data
from open_lm.model import create_model
from open_lm.distributed import init_distributed_device
from open_lm.params import parse_args

from scripts.generate_without_hf import Generator, GenerationArgs

def generate_model_jsonl(params):
    params_to_width_depth_dict = {5: (96, 3),
                    7: (128, 4),
                    9: (160, 5),
                    15: (224, 6),
                    22: (288, 8),
                    28: (320, 9),
                    37: (384, 10),
                    57: (480, 12),
                    84: (576, 14),
                    108: (640, 15),
                    149: (704, 18),
                    220: (832, 21),
                    347: (1024, 23),
                    455: (1120, 26),
                    611: (1312, 26),
                    901: (1504, 30)
                    }

    width, depth = params_to_width_depth_dict[params]
    filepath = f"layers={depth}_hidden-dim={width}.json"
    data = {
        "hidden_dim": width,
        "n_layers": depth,
        "n_heads": 4,
        "seq_len": 2048,
        "vocab_size": 50432,
        "post_embed_norm": False,
        "weight_tying": False,
        "qk_norm": True
    }

    with open(filepath, 'w') as file:
        file.write(json.dumps(data) + '\n')
    return filepath


class ModelArgs:
    def __init__(self, params, val_data, val_data_key):
        default_params = vars(parse_args(""))
        for k, v in default_params.items():
            setattr(self, k, v)
        self.model = generate_model_jsonl(params)
        self.val_data = [val_data]
        self.val_data_key = [val_data_key]
        self.per_gpu_val_batch_size = 16
        self.vocab_size = 50432
        self.seq_len = 2048
        self.wandb = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="path/to/checkpoint")

    parser.add_argument("--val-data", default="", help="Path to validation data. If empty, generate text.")
    parser.add_argument("--val-data-key", default="json.gz")

    parser.add_argument("--input-text", default="", type=str, help="Input text to generate from. If empty, evaluate on validation data.")
    parser.add_argument("--max-gen-len", default=200, type=int)
    parser.add_argument("--temperature", default=0.8, type=float)
    parser.add_argument("--top-p", default=0.95, type=float)

    args = parser.parse_args()
    params = int(re.search(r"params=(\d+)", args.checkpoint).group(1))

    checkpoint = torch.load(args.checkpoint)
    state_dict = checkpoint["state_dict"]
    state_dict = {x.replace("module.", ""): y for x, y in state_dict.items()}
    model_args = ModelArgs(params=params, val_data=args.val_data, val_data_key=args.val_data_key)
    device = init_distributed_device(model_args)
    model_args.device = device
    model = create_model(model_args)
    model.load_state_dict(state_dict)
    model.eval().cuda()
    if args.val_data != "":
        data = get_data(
            model_args,
            skip_train=True,
        )
        metrics = evaluate_loop(model, data["val_list"], 0, model_args, None)
        print(metrics)
    elif args.input_text != "":
        model = model.half()
        generator = Generator(model)
        input_text = [
            args.input_text,
        ]
        output = generator.generate(
            input_text,
            GenerationArgs(args.max_gen_len, args.temperature, args.top_p),
        )
        print("".join(output))
        
    else:
        print("Please provide either --val-data or --input-text")

if __name__ == "__main__":
    main()
