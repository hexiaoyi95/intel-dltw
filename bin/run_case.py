#!/usr/bin/env python
# encoding: utf-8

import os, sys
import argparse
import logging
from logging.config import fileConfig

os.environ['GLOG_minloglevel'] = '1'
SCRIPT_HOME = os.path.dirname(os.path.realpath(__file__))
DEFAULT_CONFIG = os.path.join(SCRIPT_HOME, '..', 'test-config', 'templates', 'image_classification_accuracy.json')

sys.path.insert(1, os.path.join(os.path.dirname(__file__), '..'))
from utils.io import json2obj
from utils.io import json2dict
from utils.io import dict2json
from applications import applications_factory

def args_process():
    arg_parser = argparse.ArgumentParser(description='')
    arg_parser.add_argument('--config', '-c', default=DEFAULT_CONFIG, help='config file for running DL Applications')
    arg_parser.add_argument('--parent_dir', '-p', default='.', help='parent dir for saving output')
    arg_parser.add_argument('--python_path', '-pp', default='', help='python path of backend')
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

    config_dict = json2dict(args.config)
    config_dict['out_dir'] = os.path.join( args.parent_dir, config_dict['out_dir'] )

    if config_dict.has_key('reference'):
        config_dict['reference']['result_dir'] = os.path.join( args.parent_dir, config_dict['reference']['result_dir'] )
    
    if args.python_path != '' :
        config_dict['backend']['python_path'] = args.python_path

    dict2json(config_dict, os.path.join( args.parent_dir, 'modified_conf.json'))
    config_modified = json2obj(os.path.join( args.parent_dir, 'modified_conf.json'))

    app = applications_factory(config_modified.application)
    app.run(config_modified)

if __name__ == "__main__":
    main()
