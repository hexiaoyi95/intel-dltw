#!/usr/bin/env python
# encoding: utf-8

import os, sys
import argparse

SCRIPT_HOME = os.path.dirname(os.path.realpath(__file__))
DEFAULT_CONFIG = os.path.join(SCRIPT_HOME, '..', 'test-config', 'config-template.json')

sys.path.insert(1, os.path.join(os.path.dirname(__file__), '..'))
from utils.io import json2obj
from applications import applications_factory
#from backends import backends_factory

def args_process():
    arg_parser = argparse.ArgumentParser(description='')
    arg_parser.add_argument('--config', '-c', default=DEFAULT_CONFIG, help='config file for running DL Applications')
    args = arg_parser.parse_args()
    return args

def main():
    args = args_process()
    config = json2obj(args.config)

    app = applications_factory(config.application)

    app.run(config)

if __name__ == "__main__":
    main()
