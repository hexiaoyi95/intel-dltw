import sys, os
import logging
import utils
from utils.result_check import check_detection_result
from utils.io import json2obj
from backends import backends_factory
import pprint

logger = logging.getLogger('root')

def test_inference_performance(backend, config):
    pass
    # logger.debug("testing performance for image classification")


    # backend.prepare_classify(config)

    # elpased_ms_list = []

    # for it in xrange(config.iteration):
    #     backend.shuffle_inputs()
    #     elapsed_ms = utils.io.benchmark.get_elapsed_ms(backend.infer)
    #     elpased_ms_list.append(elapsed_ms)

    # utils.benchmark.performance_analysis(elpased_ms_list)

def infer4result(backend, batch, config):
    backend.prepare_classify(batch, config)
    backend.infer()
    output = backend.get_detection_output()
    return output

def test_inference_accuracy(backend, config):
    logging.debug("testing accuracy for obj detection")

    input_list = utils.io.get_input_from_txt(config.input_path, config.batch_size)

    batches = utils.io.slice2batches(input_list,config.batch_size)
    total_batches = len(batches)
    outputs = {}
    count = 0
    for batch in batches:
        output = infer4result(backend, batch, config)
        outputs.update(output)
        count = count + 1
        if count % 2 == 0 or count == total_batches :
            logging.info("Done for %d/%d batches" % (count,total_batches))
    print outputs
    out_file = os.path.join(config.out_dir, 'accuracy.json')
    utils.io.dict2json(outputs, out_file)

    # if config.reference.rerun == True:
    #     ref_config = json2obj(config.reference.config)
    #     logging.debug(" rerun reference")
    #     count = 0
    #     ref_input_list = utils.io.get_input_from_txt(ref_config.input_path, ref_config.batch_size)
    #     batches = utils.io.slice2batches(ref_input_list,ref_config.batch_size)
    #     total_batches = len(batches)
    #     ref_outputs = {}
    #     ref_backend_class = backends_factory(ref_config.backend)
    #     ref_backend = ref_backend_class(ref_config)
    #     for batch in batches:
    #         output = infer4result(ref_backend, batch, ref_config)
    #         ref_outputs.update(output)
    #         count = count + 1
    #         if count % 2 == 0 or count == total_batches:
    #             logging.info("rerun reference, Done for %d/%d" % (count,total_batches))

    #     out_file = os.path.join(ref_config.out_dir, 'accuracy.json')
    #     utils.io.dict2json(ref_outputs, out_file)

    if hasattr(config, 'reference'):
        res = check_detection_result(out_file, config.reference.result_file)
        pprint.pprint(res)
        res_check_file = os.path.join(config.out_dir, 'check.json')
        utils.io.dict2json(res, res_check_file)

def run(config):
    backend_class = backends_factory(config.backend)
    backend = backend_class(config)

    if config.test_type == "performance":
        test_inference_performance(backend, config)
    elif config.test_type == "accuracy":
        test_inference_accuracy(backend, config)
    else:
        raise ValueError('Unsupported test type {}'.format(config.test_type))
