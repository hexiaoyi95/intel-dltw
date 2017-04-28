#!/usr/bin/env python
# encoding: utf-8
import os, sys
import argparse
import logging
from logging.config import fileConfig
import subprocess
import pprint

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
    logger.debug('running for reference, ref-config: {}'.format(case_conf_path))
    bin_path = os.path.join(WORKHOME, 'bin', 'run_case.py')
    ret_code = subprocess.call([
        bin_path,
        "-c",
        case_conf_path
    ])

def gen_case(suite_name, app_name, template, backend, model, batch_size):
    unique_name = "{}-{}-{}".format(backend['name'], model['name'], batch_size)
    tconf = template.copy()
    tconf['backend'] = backend
    tconf['batch_size'] = batch_size
    tconf['model'] = model
    tconf['out_dir'] = os.path.join('out', app_name, unique_name)
    if 'reference' in tconf:
        ref_unique_name = "{}-{}-{}.json".format(tconf['reference']['backend']['name'], model['name'], batch_size)
        tconf['reference']['out_file'] = os.path.join('reference', app_name, ref_unique_name, 'accuracy.json')

        #save reference as a case and run the reference case
        ref_tconf = tconf.copy()
        del ref_tconf['reference']
        ref_tconf['backend'] = tconf['reference']['backend']
        ref_tconf['out_dir'] = os.path.dirname(tconf['reference']['out_file'])
        ref_case_conf_path = os.path.join(
            WORKHOME,
            "test-config",
            suite_name,
            app_name,
            "ref",
            ref_unique_name
        )
        dict2json(ref_tconf, ref_case_conf_path)
        run_case(ref_case_conf_path)

        #check reference generated
        if not os.path.exists(tconf['reference']['out_file']):
            logger.error("reference not generated: {}".format(tconf['reference']['out_file']))
            sys.exit(1)

    # save test cases configuration
    case_conf_path = os.path.join(
        WORKHOME,
        "test-config",
        suite_name,
        app_name,
        unique_name
    )

    logger.debug(pprint.pformat(tconf))
    dict2json(tconf, case_conf_path)



def main():
    args = args_process()
    suite_config = json2dict(args.config)
    logger.debug(pprint.pformat(suite_config))
    for template_path in suite_config['app_templates']:
        logger.debug("gen cases from template: {}".format(template_path))
        template = json2dict(template_path)
        short_app_name = os.path.basename(template_path)
        short_app_name = os.path.splitext(short_app_name)[0]

        logger.debug(pprint.pformat(template))
        for model in suite_config['models']:
            for backend in suite_config['backends']:
                for batch_size in suite_config['batch_sizes']:
                    logger.debug("saving case config to {}".format(os.path.join('test-config',suite_config['name'], short_app_name)))
                    gen_case(suite_config['name'], short_app_name, template, backend, model, batch_size)


if __name__ == "__main__":
    main()
