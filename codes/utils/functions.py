import argparse
from omegaconf import OmegaConf


def setup_parser():
    # parser = argparse.ArgumentParser(add_help=False)
    # parser.add_argument('--config', type=str)
    # _args = parser.parse_args()
    # args = OmegaConf.load(_args.config)
    args = OmegaConf.load('/home/jianbo/Code/MIMIC-selector/config/wikidiverse.yaml')
    return args
