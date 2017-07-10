import sys, os
import logging
import utils
from utils.result_check import layer_accuracy_convergence
from utils.io import json2obj
from backends import backends_factory
import pprint
import numpy as np
import shutil
from collections import OrderedDict
logger = logging.getLogger('root')

def cal4result(backend, config):
    if config.model.prototxt_type == 'solver':
        backend.step(config.iteration)
    else:
        if config.model.prototxt_type == 'deploy' or config.model.prototxt_type == 'chainer_py':
            input_list = utils.io.get_input_from_txt(config.input_path)
            batches = utils.io.slice2batches(input_list, config.batch_size)
            backend.prepare_classify(batches[0], config)
        
        for i in xrange(config.iteration):
            logger.info('processing {}th forward'.format(i))
            backend.forward()
            if not config.forward_only:
                logger.info('processing {}th backward'.format(i))
                backend.backward()

    logger.debug('collecting data')
    result = backend.get_layer_accuracy_output_debug(config)

    return result


def test_layer_accuracy(backend, config):

    logging.debug("testing layer accuracy")

    check_result = list()
    result = cal4result(backend, config)
    if os.path.exists(config.out_dir):
        shutil.rmtree(config.out_dir)
    for layer_name, l in result.iteritems():
        for j,[blob_name, np_list] in enumerate(l):
            for i, np_arry in enumerate(np_list):
                if blob_name == 'params_diff':
                    ctx = 'params_{}_diff'.format(i)
                elif blob_name == 'params_data':
                    ctx = 'params_{}_data'.format(i)
                else:
                    if i == 0:
                        ctx = blob_name + '_data'
                    else:
                        ctx = blob_name + '_diff'

                np_name = os.path.join(config.out_dir, layer_name.replace('/','-'), blob_name + '_' + ctx)
                np_name = os.path.expanduser(np_name)
                if not os.path.exists(os.path.dirname(np_name)):
                    os.makedirs(os.path.dirname(np_name))
                np.save(np_name, np_arry)
                with open(np_name + ".txt", 'w') as f:
                    f.write(str(np_arry))

    if hasattr(config,'reference'):
        result_dir = os.path.expanduser(config.reference.result_dir)
        this_batch_result = layer_accuracy_convergence(backend, result,config.out_dir, result_dir, config, config.precision)
        check_result.extend(this_batch_result)


    if hasattr(config,'reference'):
        with open(os.path.join(config.out_dir, 'test_report.txt'),'w') as fp:
            for line in check_result:
                for word in line:
                    #print
                    fp.write(str(word))
                    if type(word) == type('?') and  word == '-' :
                        continue
                    fp.write('\t')
                fp.write('\n')
            fp.write('\n')
            fp.write('\n')


def run(config):
    backend_class = backends_factory(config.backend)
    backend = backend_class(config)

    test_layer_accuracy(backend, config)
