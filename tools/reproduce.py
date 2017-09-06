#!/usr/bin/env python
# encoding: utf-8
import sys
import os
import importlib
import argparse
import logging
import numpy as np

os.environ['glog_minloglevel'] = '1'
workhome = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(1, workhome)

from utils.io import json2obj
from utils import result_check

logger = logging.getLogger('root')
def args_process():
    arg_parser = argparse.ArgumentParser(description='')
    arg_parser.add_argument('--config', '-c', default="", help='config file for reproduce')
    arg_parser.add_argument('--topology', '-t', default="", help='topology for reproduce')
    args = arg_parser.parse_args()
    return args



def main():
    args = args_process()


    config = json2obj(args.config)
    ref_config = json2obj(str(config.ref_config)) 
    sys.path.insert(1, os.path.expanduser(config.python_path))
     
    caffe  = __import__("caffe")
    caffe.set_mode_cpu()
    caffe.set_random_seed(0) 
    
    if config.phase == 'test':
        phase = caffe.TEST
    else:
        phase = caffe.TRAIN
    #config.ref_bottom_data_path, config.bottom_data_path
    input_data = [config.ref_bottom_data_path, config.ref_bottom_data_path]

    ref_engine = str(ref_config.backend.engine) if ref_config.backend.engine != 'default' \
                    else 'CAFFE'
    test_engine = str(config.engine) if config.engine != 'default' else 'CAFFE'

    net_ref = caffe.Net(str(args.topology), phase, weights=str(config.weight), engine=ref_engine)
    net_test = caffe.Net(str(args.topology), phase, weights=str(config.weight), engine=test_engine)
     
    net = [net_ref, net_test]
    
    output_data = list()
    for index,n in enumerate(net):
        bottom_data = input_data[index]
        for blob_name, blob_data_path in bottom_data:
            n.blobs[blob_name].data[...] = np.load(blob_data_path)
        n.forward()
        top_data = dict()
        for out in n.outputs:
            top_data[out] = n.blobs[out].data
        output_data.append(top_data)
    
    target_top_blob_name = config.top_data_path[0]

    
    check_result, detail =  result_check.check_error(output_data[0][target_top_blob_name], \
                                                     output_data[1][target_top_blob_name], \
                                                     target_top_blob_name, \
                                                     config.precision.rtol, \
                                                     config.precision.atol,\
                                                     config.precision.check_method) 
    print check_result
    for line in detail:
        print line

if __name__ == '__main__':
    main()
