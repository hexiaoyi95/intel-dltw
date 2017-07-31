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
    arg_parser.add_argument('--run_ref', '-r', default='off', help='whether rerun reference')
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
    #[[json_path,is_ref],[],...]
    jsonPathList = io.genConfFilename(args.config)
    report_path_txt = os.path.join('test-config-debug',os.path.splitext(os.path.basename(args.config))[0] + '.txt' )

    if not os.path.exists(args.parent_dir):
        os.mkdir(args.parent_dir)
 
    call(["mv",report_path_txt, os.path.join(args.parent_dir,'find_the_report.txt')])
    
    report_path_list=list()
    raw_lines = list()
    if args.run_ref == 'on':
        shutil.rmtree(args.parent_dir)
    else:
        with open(os.path.join(args.parent_dir,'find_the_report.txt')) as fp:
            #skip the first line(title),get report path
            raw_lines = fp.readlines()
            report_path_list = [ x.split('\t')[-1].strip() for x in raw_lines[1:]]
            for report_path in report_path_list:         
                real_path = os.path.join( args.parent_dir, os.path.dirname(report_path))
                if os.path.exists(real_path):
                    shutil.rmtree(real_path)
    
    for jsonPath,is_ref in jsonPathList:
        if is_ref and args.run_ref != 'on':
            continue
        call(["./bin/run_case.py", "-c", jsonPath, "-p", args.parent_dir, "-pp", args.python_path])
    test_case_successed = 0
    test_case_failed = 0
    with open(os.path.join(args.parent_dir,'test_result.txt'),'w') as test_result_fp:
        for index,line in enumerate(raw_lines):    
            if index == 0:
                new_line = line.strip() + '\t' + 'pass/fail'
                test_result_fp.write(new_line)
                test_result_fp.write('\n')
            else:
                report_path = line.split('\t')[-1].strip()
                try:
                    report_fp = open(os.path.join(args.parent_dir,report_path))
                except:
                    test_case_failed += 1
                else:
                    test_case_successed +=1
                    pass_or_fail = report_fp.readline().strip().split('\t')[-1]
                    new_line = line.strip() + '\t' + pass_or_fail
                    test_result_fp.write(new_line)
                    test_result_fp.write('\n')
                    report_fp.close()
    
if __name__ == '__main__':
    main()
