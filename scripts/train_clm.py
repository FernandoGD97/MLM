#!/usr/bin/env python
import argparse
from encoder_pretraining.training.factory import run_from_config
def main():
    p=argparse.ArgumentParser(description="Same-encoder causal continual pretraining with per-epoch probes")
    p.add_argument("--config",required=True); args=p.parse_args(); run_from_config(args.config,"clm")
if __name__=="__main__":main()
