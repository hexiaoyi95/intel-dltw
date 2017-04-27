#!/usr/bin/env python
# encoding: utf-8
import os, sys
import argparse
import logging
from logging.config import fileConfig
import subprocess
import pprint
import shutil

os.environ['GLOG_minloglevel'] = '1'
WORKHOME = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(1, WORKHOME)
from utils.io import  json2dict, dict2json

DEFAULT_CONFIG = os.path.join(WORKHOME, 'test-config', 'templates', 'test-suite.json')

fileConfig(os.path.join(WORKHOME, 'logging_config.ini'))
logger = logging.getLogger('root')

def args_process():
    arg_parser = argparse.ArgumentParser(description='')
    arg_parser.add_argument('--config', '-c', default=DEFAULT_CONFIG, help='config file for test suite')
    args = arg_parser.parse_args()
    return args

def run_case(case_conf_path):
    logger.info('[run] {}'.format(case_conf_path))
    bin_path = os.path.join(WORKHOME, 'bin', 'run_case.py')
    ret_code = subprocess.call([
        bin_path,
        "-c",
        case_conf_path
    ])
    if 0 != ret_code:
        logger.error("{} -c {}", bin_path, case_conf_path)

def pretty_result(result):
    """
    Input: result [True, False]

    convert: True  -> pass
             False -> fail

    Return [pass, fail]
    """
    if not result:
        return ['crash','crash']

    for i in xrange(len(result)):
        if result[i]:
            result[i] = 'pass'
        else:
            result[i] = 'fail'
    return result


def collect_result(result_dir, topN):
    result_file = os.path.join(result_dir, "check.json")
    logger.info('[collect] {}'.format(result_dir))
    try:
        result = json2dict(result_file)
    except:
        return ["crash", "crash"]

    return pretty_result(result[topN])


def main():
    args = args_process()
    suite_config = json2dict(args.config)
    logger.debug(pprint.pformat(suite_config))
    suite_name = suite_config['name']
    results = {}
    topN = "top_3"
    for template_path in suite_config['app_templates']:
        short_app_name = os.path.basename(template_path)
        short_app_name = os.path.splitext(short_app_name)[0]
        logger.debug("Start runing case in: {}".format(os.path.join('test-config',suite_config['name'], short_app_name)))

        for model in suite_config['models']:
            if not model['name'] in results:
                results[model['name']] = {}

            for backend in suite_config['backends']:
                if not backend['name'] in results[model['name']]:
                    results[model['name']][backend['name']] = {}

                for batch_size in suite_config['batch_sizes']:
                    unique_name = "{}-{}-{}.json".format(backend['name'], model['name'], batch_size)
                    case_conf_path = os.path.join(
                        WORKHOME,
                        'test-config',
                        suite_name,
                        short_app_name,
                        unique_name
                    )
                    case_conf = json2dict(case_conf_path)
                    if os.path.exists(case_conf['out_dir']):
                        shutil.rmtree(case_conf['out_dir'])

                    run_case(case_conf_path)

                    result = collect_result(case_conf['out_dir'], topN)
                    results[model['name']][backend['name']][batch_size] = result

        pprint.pprint(results)

if __name__ == "__main__":
    main()
