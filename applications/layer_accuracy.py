import sys, os
import logging
import utils
from utils.result_check import check_layer_accuracy_result
from utils.io import json2obj
from backends import backends_factory
import pprint
import numpy as np
logger = logging.getLogger('root')

def cal4result(backend, batch, config):
    backend.prepare_classify(batch, config)
    backend.infer()
    backend.backward();
    datas, weights = backend.get_layer_accuracy_output()

    return datas,weights



def test_layer_accuracy(backend, config):

    logging.debug("testing layer accuracy")
    input_list = utils.io.get_input_from_txt(config.input_path)


    batches = utils.io.slice2batches(input_list, config.batch_size)
    total_batches = len(batches)
    outputs = {}
    batches_name = {}
    count = 0
    check_result = [True,True]
    for batch in batches:
        datas, weights = cal4result(backend, batch, config)
        count = count + 1
        result = {'datas': datas, 'weights': weights}
        batch_name = { str(count): batch}
        batches_name.update(batch_name)
        if config.isref == True:
            for data_type, data in result.iteritems():
                for key, value in data.iteritems():
                    npy_path = config.out_dir + '/' + str(count) + '/' + key + '_' + data_type
                    npy_path = os.path.expanduser(npy_path)
                    if not os.path.exists(os.path.dirname(npy_path)):
                        os.makedirs(os.path.dirname(npy_path))
                    np.save(npy_path, value)
        else:
            result_dir = os.path.expanduser(config.reference.result_dir)
            check_result = check_layer_accuracy_result(batch_name, datas, weights, result_dir, check_result)
            pprint.pprint(check_result)


        if count % 2 == 0 or count == total_batches:
            logging.info("Done for %d/%d batches" % (count,total_batches))


    name_file =  os.path.join(config.out_dir, 'name.json')
    utils.io.dict2json(batches_name, name_file)

    pprint.pprint(check_result)
    res_check_file = os.path.join(config.out_dir, 'check.json')
    utils.io.dict2json(check_result, res_check_file)

def run(config):
    backend_class = backends_factory(config.backend)
    backend = backend_class(config)

    if config.test_type == "performance":
        test_layer_performance(backend, config)
    elif config.test_type == "accuracy":
        test_layer_accuracy(backend, config)
    else:
        raise ValueError('Unsupported test type {}'.format(config.test_type))
