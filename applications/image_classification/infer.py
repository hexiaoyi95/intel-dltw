import sys, os
import logging
import utils
from utils.result_check import check_classify_result
from backends import backends_factory
import pprint

logger = logging.getLogger('root')

def test_inference_performance(backend, config):
    logger.debug("testing performance for image classification")


    backend.prepare_classify(config)

    elpased_ms_list = []

    for it in xrange(config.iteration):
        backend.shuffle_inputs()
        elapsed_ms = utils.io.benchmark.get_elapsed_ms(backend.infer)
        elpased_ms_list.append(elapsed_ms)

    utils.benchmark.performance_analysis(elpased_ms_list)

def infer4result(backend, batch, config):
    backend.prepare_classify(batch, config)
    backend.infer()
    output = backend.get_classify_output()
    return output

def test_inference_accuracy(backend, config):
    logging.debug("testing performance for image classification")

    input_list = utils.io.get_input_list(config.input_path, config.batch_size)
    batches = utils.io.slice2batches(input_list,config.batch_size)
    outputs = {}
    for batch in batches:
        output = infer4result(backend, batch, config)
        outputs.update(output)

    out_file = os.path.join(config.out_dir, 'accuracy.json')
    utils.io.dict2json(outputs, out_file)

    if hasattr(config, 'reference'):
        if not os.path.exists(config.reference.out_file):
            logger.info('reference not found: {}'.format(config.reference.out_file))
            return
        res = check_classify_result(out_file, config.reference.out_file)
        logger.debug(pprint.pformat(res))
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
