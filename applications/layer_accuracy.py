import sys, os
import logging
import utils
from utils.result_check import check_layer_accuracy_result
from utils.io import json2obj
from backends import backends_factory
import pprint
import numpy as np
from collections import OrderedDict
logger = logging.getLogger('root')

def cal4result(backend, batch, config):
    backend.prepare_classify(batch, config)
    backend.infer()
    backend.backward();
    datas, diffs = backend.get_layer_accuracy_output()
    #result = backend.get_layer_accuracy_output()

    return datas,diffs
    #return result


def test_layer_accuracy(backend, config):

    logging.debug("testing layer accuracy")
    input_list = utils.io.get_input_from_txt(config.input_path)


    batches = utils.io.slice2batches(input_list, config.batch_size)
    total_batches = len(batches)

    batches_name = {}
    count = 0

    check_result = OrderedDict()
    for batch in batches:
        datas, diffs = cal4result(backend, batch, config)
        count = count + 1
        result = {'datas': datas, 'diffs': diffs}
        #result = cal4result(backend, batch, config)
        batch_name = { str(count): batch}
        batches_name.update(batch_name)

        if hasattr(config,'reference'):
            result_dir = os.path.expanduser(config.reference.result_dir)
            check_result = check_layer_accuracy_result(batch_name, datas, diffs, result_dir, check_result)
            #check_result = check_layer_accuracy_result(count, batch, result,result_dir, check_result)

        else:
            for data_type, data in result.iteritems():
                for key, value in data.iteritems():
                    npy_path = config.out_dir + '/' + str(count) + '/' + key + '_' + data_type
                    npy_path = os.path.expanduser(npy_path)
                    if not os.path.exists(os.path.dirname(npy_path)):
                        os.makedirs(os.path.dirname(npy_path))
                    np.save(npy_path, value)

            # for layer_name, l in result.iteritems():
            #     for blob_name, np_list in l:
            #         for i, np_arry in enumerate(np_list):
            #             np_name = os.path.join(config.out_dir, str(count), layer_name, blob_name + '_' + str(i))
            #             np_name = os.path.expanduser(np_name)
            #             if not os.path.exists(os.path.dirname(np_name)):
            #                 os.makedirs(os.path.dirname(np_name))
            #             np.save(np_name, np_arry)

        if count % 2 == 0 or count == total_batches:
            logging.info("Done for %d/%d batches" % (count,total_batches))


    name_file =  os.path.join(config.out_dir, 'name.json')
    utils.io.dict2json(batches_name, name_file)

    # pprint.pprint(check_result)
    # print check_result
    # with open(os.path.join(config.out_dir, 'layer_accuracy check.txt'),'w') as fp:
    #     for line in check_result:
    #         for word in line:
    #             print word
    #             fp.write(str(word))
    #             if word != '-' and ':' is not in word:
    #                 fp.write('\t')
    #         fp.write('\n')
    #     fp.write('\n')
    #     fp.write('\n')


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
