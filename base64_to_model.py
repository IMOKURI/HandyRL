#!/usr/bin/env python

import argparse
import base64
import bz2
import importlib
import pickle

import torch


def main():
    args = get_args()

    with open(args.model_path[0], 'r') as f:
        model_string = f.read().strip()[2:-1].encode()

    state_dict = pickle.loads(bz2.decompress(base64.b64decode(model_string)))

    hungry_geese = importlib.import_module("handyrl.envs.kaggle.hungry_geese")
    env = hungry_geese.Environment()
    model = hungry_geese.GeeseNet(env)

    model.load_state_dict(state_dict)

    out = args.model_path[0].replace(".txt", ".pth")
    torch.save(model.state_dict(), out)


def get_args():
    parser = argparse.ArgumentParser(description="""
    Convert base64 string to model.
    """)

    parser.add_argument('--model-path',
                        '-m',
                        nargs=1,
                        required=True,
                        help='Model path')

    return parser.parse_args()


if __name__ == "__main__":
    main()
