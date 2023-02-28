import os
import json
import torch
import shutil
import argparse

from transformers import AutoTokenizer
from modeling_splade import SpladeModel, SpladeConfig

# python /home/grads/r/rohan.chaudhury/multidoc2dial/multidoc2dial/md2d/rag_splade/convert_from_beir_to_transformers.py --checkpoint_dir /home/grads/r/rohan.chaudhury/multidoc2dial/multidoc2dial/splade/distilsplade_max --dump_path $CHECKPOINTS/rag-distilsplade


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir")
    parser.add_argument("--dump_path")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = get_args()

    os.makedirs(args.dump_path, exist_ok=True)

    for file in os.listdir(args.checkpoint_dir):
        print(file)
        if "pytorch_model.bin" in file:
            old_checkpoint = torch.load(os.path.join(args.checkpoint_dir, "pytorch_model.bin"), map_location='cpu')
            new_checkpoint = {"auto_model." + k : v for k,v in old_checkpoint.items()}
            with open(os.path.join(args.dump_path, "pytorch_model.bin"), 'wb') as fi:
                torch.save(new_checkpoint, fi)
        elif "config.json" in file:
            config = SpladeConfig()
            with open(os.path.join(os.path.join(args.checkpoint_dir, "config.json")), 'r') as fi:
                previous_config = json.load(fi)
            for k,v in previous_config.items():
                if hasattr(config, k):
                    if k != "model_type":
                        setattr(config,k, v)
                    else:
                        print("skipping model_type")
                else:
                    print(f"key: {k} not present in config!")
            config.save_pretrained(args.dump_path)
        else:
            shutil.copy(os.path.join(args.checkpoint_dir, file), args.dump_path)
        
    print("loading model weights for sanity")
    model = SpladeModel.from_pretrained(args.dump_path)
    config = SpladeConfig.from_pretrained(args.dump_path)
    tokenizer = AutoTokenizer.from_pretrained(args.dump_path)
    print("sane!")