#!/usr/bin/env python

import argparse
import base64
import bz2
import pickle

import torch


def main():
    args = get_args()

    state_dict = torch.load(args.model_path[0])

    model_b64 = base64.b64encode(bz2.compress(pickle.dumps(state_dict)))

    out = args.model_path[0].replace(".pth", ".txt")
    with open(out, "w") as f:
        f.write(str(model_b64))


def get_args():
    parser = argparse.ArgumentParser(description="""
    Convert model to base64 string.
    """)

    parser.add_argument('--model-path',
                        '-m',
                        nargs=1,
                        required=True,
                        help='Model path')

    return parser.parse_args()


if __name__ == "__main__":
    main()
