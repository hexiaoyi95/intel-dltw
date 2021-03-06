#!/usr/bin/env python
# encoding: utf-8
import sys
import os 
import logging
import argparse
import shutil
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
    arg_parser.add_argument('--parent_dir', '-p', default='core_out', help='parent dir for saving output')
    arg_parser.add_argument('--run_ref', '-r', default='off', help='only run reference')
    arg_parser.add_argument('--python_path', '-pp', default='', help='python path of backend')
    arg_parser.add_argument('--cpu_type', '-cpu', default='core', help='target cpu type')
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

def modify_conf(case_info_dict):
    if case_info_dict['prototxt_type'] == 'train_val': 
        if case_info_dict['forward_only']:
            case_info_dict['accuracy_level'] = 'fwd'
        else:
            case_info_dict['accuracy_level'] = 'bwd'
        case_info_dict.pop('prototxt_type')
        case_info_dict.pop('forward_only')
    elif case_info_dict['prototxt_type'] == 'solver':
        case_info_dict.pop('prototxt_type')
        case_info_dict['accuracy_level'] = 'train'

    if case_info_dict['engine'] == 'default':
        case_info_dict['engine'] = 'CAFFE'


    return case_info_dict
    
def main():
    args = args_process()

    setup_logger()
    #[[json_path,is_ref],[],...]
    jsonPathList = io.genConfFilename(args.config)
    report_path_txt = os.path.join('test-config-debug',os.path.splitext(os.path.basename(args.config))[0] + '.txt' )

    cases_info_json = os.path.join('test-config-debug',os.path.splitext(os.path.basename(args.config))[0] + '_cases_info.json' )

    if not os.path.exists(args.parent_dir):
        os.makedirs(args.parent_dir)
 
    
    report_path_list=list()
    raw_lines = list()
    if args.run_ref == 'on':
        shutil.rmtree(args.parent_dir)
        os.makedirs(args.parent_dir)
    
    call(["cp",report_path_txt, os.path.join(args.parent_dir,'find_the_report.txt')])
    
    with open(os.path.join(args.parent_dir,'find_the_report.txt')) as fp:
        #skip the first line(title),get report path
        raw_lines = fp.readlines()
        report_path_list = [ x.split('\t')[-1].strip() for x in raw_lines[1:]]
        for report_path in report_path_list:         
            real_path = os.path.join( args.parent_dir, os.path.dirname(report_path))
            if os.path.exists(real_path):
                shutil.rmtree(real_path)
    
    for jsonPath,is_ref in jsonPathList:
        if ( args.run_ref != 'on' and not is_ref ) or (args.run_ref == 'on' and is_ref ):
            call(["./bin/run_case.py", "-c", jsonPath, "-p", args.parent_dir, "-pp", args.python_path])
        else:
            continue
   
    test_case_successed = 0
    test_case_failed = 0

    if os.path.exists(os.path.join(args.parent_dir,'test_report')):
        shutil.rmtree(os.path.join(args.parent_dir,'test_report'))
    os.makedirs(os.path.join(args.parent_dir,'test_report'))

  #  with open(os.path.join(args.parent_dir,'test_result.txt'),'w') as test_result_fp:
  #      for index,line in enumerate(raw_lines):    
  #          if index == 0:
  #              new_line = line.strip() + '\t' + 'pass/fail'
  #              test_result_fp.write(new_line)
  #              test_result_fp.write('\n')
  #          else:
  #              report_path = line.split('\t')[-1].strip()
  #              try:
  #                  report_fp = open(os.path.join(args.parent_dir,report_path))
  #              except:
  #                  test_case_failed += 1
  #              else:
  #                  call(["cp", os.path.join(args.parent_dir,report_path), \
  #                      os.path.join(args.parent_dir,'test_report',os.path.dirname(report_path) + '.txt')])
  #                  test_case_successed +=1
  #                  pass_or_fail = report_fp.readline().strip().split('\t')[-1]
  #                  new_line = line.strip() + '\t' + pass_or_fail
  #                  test_result_fp.write(new_line)
  #                  test_result_fp.write('\n')
  #                  report_fp.close()

    cases_info = io.json2dict(cases_info_json)
    cases_info['cpu_type'] = args.cpu_type
    if cases_info['application'] == 'accuracy':
        cases_info_list = cases_info['cases_info']
        for case_info_dict in cases_info_list:
            
            #find test result
            if case_info_dict.has_key('report_path'):
                report_path = case_info_dict['report_path'] 
                try:
                    report_fp = open(os.path.join(args.parent_dir,report_path))
                except:
                    case_info_dict['test_result'] = 'cannot find test report' 
                    case_info_dict['report_path'] = ''
                else:
                    call(["cp", os.path.join(args.parent_dir,report_path), \
                        os.path.join(args.parent_dir,'test_report',os.path.dirname(report_path) + '.txt')])
                    test_case_successed +=1
                    pass_or_fail = report_fp.readline().strip().split('\t')[-1]
                    case_info_dict['test_result'] = pass_or_fail
                    case_info_dict['report_path'] = os.path.join(args.parent_dir, 'test_report',\
                        os.path.dirname(report_path) + '.txt')
                    report_fp.close()
            #redefine  
            case_info_dict = modify_conf(case_info_dict)
                
    io.dict2json(cases_info, os.path.join(args.parent_dir,'test_results.json')) 
    
if __name__ == '__main__':
    main()
