import sys, os
import logging
import utils
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
    #ref_backend = backends_factory(config.ref_backend)

    #ref_output = infer4result(ref_backend, config)

    #print ref_output
    return

def run(config):
    backend_class = backends_factory(config.backend)
    backend = backend_class(config)

    if config.test_type == "performance":
        test_inference_performance(backend, config)
    elif config.test_type == "accuracy":
        test_inference_accuracy(backend, config)
    else:
        raise ValueError('Unsupported test type {}'.format(config.test_type))
