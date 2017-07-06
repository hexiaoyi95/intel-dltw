#!/usr/bin/env python
# encoding: utf-8
import sys
import os 
import logging
import argparse
from logging.config import fileConfig
from subprocess import call


os.environ['GLOG_minloglevel'] = '1'
SCRIPT_HOME = os.path.dirname(os.path.realpath(__file__))
DEFAULT_CONFIG = os.path.join(SCRIPT_HOME, '..', 'test-config', 'templates', 'image_classification_accuracy.json')

sys.path.insert(1, os.path.join(os.path.dirname(__file__), '..'))
from utils import io

def args_process():
    arg_parser = argparse.ArgumentParser(description='')
    arg_parser.add_argument('--config', '-c', default=DEFAULT_CONFIG, help='config file for running suite')
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

    jsonPathList = io.genConfFilename(args.config)
    
    for jsonPath in jsonPathList:
        print jsonPath
        call(["./bin/run_case.py", "-c", jsonPath])
    

if __name__ == '__main__':
    main()
