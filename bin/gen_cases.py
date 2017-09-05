#!/usr/bin/env python
# encoding: utf-8
import sys
import os
import argparse
import logging
from logging.config import fileconfig


os.environ['glog_minloglevel'] = '1'
workhome = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(1, workhome)
from utils import io

DEFAULT_CONFIG = os.path.join(WORKHOME, 'test-config', 'templates', 'test-gen-cases.json')

fileConfig(os.path.join(WORKHOME, 'logging_config.ini'))
logger = logging.getLogger('root')

def args_process():
    arg_parser = argparse.ArgumentParser(description='')
    arg_parser.add_argument('--config', '-c', default=DEFAULT_CONFIG, help='config file for gen cases')
    args = arg_parser.parse_args()
    return args

def setup_logger():
    fileConfig(os.path.join(WORKHOME, 'logging_config.ini'))

def genBackend( python_path, backend, engine):
    if backend == 'caffe':
        class_path = "backends.caffe_backend.CaffeBackend"
    elif backend == 'chainer':
        class_path = "backends.chainer_backend.ChainerBackend"
    else:
        raise Exception("unsupported backend, choose caffe or chainer")
    
    return {'python_path':python_path, 'class_path': class_path, 'engine': engine}

def genModel(topology_name, prototxt_type, weight, backend):
    result = dict() 
    if backend == 'caffe':
        if prototxt_type == 'train_val':
            prototxt_name = 'img_train_val.prototxt'
            caffe_type = 'train'
        elif prototxt_type == 'deploy':
            prototxt_name = 'deploy.prototxt'
            caffe_type = 'test'
        elif prototxt_type == 'solver':
            prototxt_name = 'solver.prototxt'
            caffe_type = 'train'
        else:
            raise Exception("unsupported prototxt type, choose train_val, deploy or solver")
        if caffe_type != None:
            result['type'] =  caffe_type
    elif backend == 'chainer':
        prototxt_name = topology_name + '.py'
        prototxt_type = 'chainer_py'

    
    result['topology'] = os.path.join('dl-models',topology_name, prototxt_name)
    result['prototxt_type'] = prototxt_type
            
    
    if weight == 'default':
        if backend == 'caffe':
            weight_path = os.path.join('dl-models', topology_name, topology_name + '.caffemodel')
        elif backend == 'chainer':
            weight_path = os.path.join('dl-models', topology_name, topology_name + '.chainermodel')
            
    else:
        weight_path = weight
    
    if weight_path != None:
        result['weight'] = weight_path
    

    return result        
    

def genJson(json_name,
             topology,
             python_path,
             backend,
             engine,
             prototxt_type,
             out_dir,
             application,
             iteration,
             batch_size,
             reference_dir = None,
             forward_only = False,
             weight = None,
             mean_value = None,
             input_path = None,
             precision = None,
             reportOrder = None):
    _backend = genBackend( python_path, backend, engine)
    _model = genModel( topology, prototxt_type, weight, backend)
    
    result = dict()
    result['backend'] = _backend
    result['model'] = _model
    result['application'] = "applications." + application
    result['forward_only'] = forward_only
    result['iteration'] = int(iteration)
    result['out_dir'] = out_dir
    result['batch_size'] = int(batch_size)
    if input_path != None:
        result['input_path'] = input_path
    if mean_value != None:
        result['mean_value'] = mean_value
    if reference_dir != None:
        result['reference'] = {'result_dir' : reference_dir}
    if precision != None:
        result['precision'] = precision
    if reportOrder != None:
        result['getReport'] = {'reportOrder': reportOrder} 
    print json_name
    io.dict2json( result, json_name)

def main():
   
   args = args_process()

   setup_logger()
   
   result = io.genConfFilename(args.config,getJson_only = False)
   
   for [jsonPath, changed_template] in result:
       #print jsonPath
       genJson(jsonPath, **changed_template) 
        
   #confDict = io.json2dict(sys.argv[1])
   # [ref_conf, ref_confValue, app, confPermutations ] = io.genConfFilename(sys.argv[1])
   # template = io.json2dict('test-config/' + app + '-template.json')
   # for confList in confPermutations:
   #     #print ref_conf,ref_confValue, confList
   #     ref_dir = app
   #     out_dir = app 
   #     is_ref = False
   #     for [confName, value] in confList:
   #         template[confName] = value
   #         out_dir += '_' + value
   #         if confName == ref_conf:
   #             ref_dir += '_' + ref_confValue
   #             if ref_conf == value:
   #                 is_ref = True
   #         else: 
   #             ref_dir += '_' + value
   #     if !is_ref:
   #         template['reference_dir'] = 'out/' + ref_dir
   #     template['out_dir'] = 'out/' + out_dir
   #     jsonPath = 'test-config/' + out_dir + '.json'
   #     genJson( jsonPath, **template)
   #     

   # app = confDict['application']
   # with open('run_case.sh','w') as fp:
   #     fp.write('#!/bin/bash\n')
   #     template = io.t( 'test-config/' + app + '-template.json')
   #     for item, argDict in confDict.iteritems():
   #         if item != 'application':
   #             reference_dir = ''
   #             for i, arg in argDict.iteritems():
   #                 template[item] = arg
   #                 if i == 'ref':
   #                     reference_dir = 'out/' + item +'/ref'
   #                     template['out_dir'] = reference_dir
   #                     template['reference_dir'] = reference_dir
   #                 else:
   #                     template['out_dir'] = 'out/' + item + '/' + arg
   #                     template['reference_dir'] = reference_dir
   #                 genJson('test-config/' + app + '_' +  item + '_' + arg + '.json', **template)
   #                 fp.write('./bin/run_cafe -c ' + 'test-config/' + app + '_' +  item + '_' + arg + '.json\n')

if __name__ == '__main__':
    main()
