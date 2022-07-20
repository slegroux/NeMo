#!/usr/bin/env python
import logging
logging.disable(logging.CRITICAL)
import argparse
from pathlib import Path
logging.getLogger('nemo_logger').setLevel(logging.CRITICAL)

def get_best_ckpt_from_last_run(base_dir, checkpoints_dir, model_name='FastPitch'):
    exp_dirs = list([i for i in (Path(base_dir) / checkpoints_dir / model_name).iterdir() if i.is_dir()])
    last_exp_dir = sorted(exp_dirs)[-1]
    last_checkpoint_dir = last_exp_dir / "checkpoints"
    last_ckpt = list(last_checkpoint_dir.glob('*-last.ckpt'))
    if len(last_ckpt) == 0:
        raise ValueError(f"There is no last checkpoint in {last_checkpoint_dir}.")
    return str(last_ckpt[0])


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", default='.')
    parser.add_argument("--checkpoint_dir")
    parser.add_argument("--model_name", default='FastPitch')
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    last_ckpt = get_best_ckpt_from_last_run(args.base_dir, args.checkpoint_dir, model_name=args.model_name)
    print(last_ckpt)