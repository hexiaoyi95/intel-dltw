#!/usr/bin/env python
# encoding: utf-8

import os, sys
import argparse
import logging
from logging.config import fileConfig

# os.environ['GLOG_minloglevel'] = '3'
SCRIPT_HOME = os.path.dirname(os.path.realpath(__file__))
DEFAULT_CONFIG = os.path.join(SCRIPT_HOME, '..', 'test-config', 'templates', 'img-classification-infer-accuracy.json')

sys.path.insert(1, os.path.join(os.path.dirname(__file__), '..'))
from utils.io import json2obj
from applications import applications_factory

def args_process():
    arg_parser = argparse.ArgumentParser(description='')
    arg_parser.add_argument('--config', '-c', default=DEFAULT_CONFIG, help='config file for running DL Applications')
    args = arg_parser.parse_args()
    return args

def setup_logger():
    fileConfig(os.path.join(SCRIPT_HOME,'..', 'logging_config.ini'))
    # formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

    # handler = logging.StreamHandler()
    # handler.setFormatter(formatter)

    # logger = logging.getLogger(name)
    # logger.setLevel(logging.DEBUG)
    # logger.addHandler(handler)

    # return logging.getLogger(name)

def main():
    args = args_process()

    setup_logger()

    config = json2obj(args.config)

    app = applications_factory(config.application)
    app.run(config)

if __name__ == "__main__":
    main()
