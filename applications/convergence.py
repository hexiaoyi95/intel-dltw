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

    for i in xrange(config.iteration):
        logger.debug('processing {}th forward'.format(i))

        # print backend.net.blobs['data'].data
        backend.forward()
	#backend.get_layer_accuracy_output_debug()        
	# print backend.net.blobs['data'].data
        logger.debug('processing {}th backward'.format(i))
        #backend.backward()
    #datas, diffs = backend.get_layer_accuracy_output()
    logger.debug('collecting data')
    result = backend.get_layer_accuracy_output_debug() 
    
    #return datas,diffs
    return result


def test_layer_accuracy(backend, config):

    logging.debug("testing layer accuracy")

    check_result = list()
    result = cal4result(backend, config)

    for layer_name, l in result.iteritems():
        for blob_name, np_list in l:
            for i, np_arry in enumerate(np_list):
                if blob_name == 'params_diff':
                    if i == 0:
                        ctx = 'W'
                    else:
                        ctx = 'b'
                else:
                    if i == 0:
                        ctx = 'data'
                    else:
                        ctx = 'diff'

                np_name = os.path.join(config.out_dir, layer_name.replace('/','-'), blob_name + '_' + ctx)
                np_name = os.path.expanduser(np_name)
                if not os.path.exists(os.path.dirname(np_name)):
                    os.makedirs(os.path.dirname(np_name))
                np.save(np_name, np_arry)
                with open(np_name + ".txt", 'w') as f:
                    f.write(str(np_arry))

    if hasattr(config,'reference'):
        result_dir = os.path.expanduser(config.reference.result_dir)
        #check_result = layer_accuracy_debug(batch_name, datas, diffs, result_dir, check_result, config.precision)
        this_batch_result = layer_accuracy_convergence(backend, result,result_dir, config.precision)
        check_result.extend(this_batch_result)


    # name_file =  os.path.join(config.out_dir, 'name.json')
    # utils.io.dict2json(batches_name, name_file)

    #pprint.pprint(check_result)
    #print check_result
    if hasattr(config,'reference'):
        with open(os.path.join(config.out_dir, 'layer_accuracy_Report'),'w') as fp:
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


    #res_check_file = os.path.join(config.out_dir, 'check.json')
    #utils.io.dict2json(check_result, res_check_file)

def run(config):
    backend_class = backends_factory(config.backend)
    backend = backend_class(config)

    test_layer_accuracy(backend, config)
