import sys, os
import logging
from utils.benchmark import get_elapsed_ms, performance_analysis
from backends import backends_factory

logger = logging.getLogger('root')

def test_inference_performance(backend, config):
    logger.debug("testing performance for image classification")


    backend.prepare_classify(config)

    elpased_ms_list = []

    for it in xrange(config.iteration):
        backend.shuffle_inputs()
        elapsed_ms = get_elapsed_ms(backend.infer)
        print elapsed_ms
        elpased_ms_list.append(elapsed_ms)

    performance_analysis(elpased_ms_list)

def infer4result(backend, config):
    backend.prepare_classify(config)
    backend.infer()
    output = backend.get_classify_output()
    return output

def test_inference_accuracy(backend, config):
    logging.debug("testing performance for image classification")

    output = infer4result(backend, config)

    # ref_backend = backends_factory(config.ref_backend)

    #ref_output = infer4result(ref_backend, config)

    print output
    #print ref_output
    return

def run(config):
    backend_class = backends_factory(config.backend)
    backend = backend_class()

    if config.test_type == "performance":
        test_inference_performance(backend, config)
    elif config.test_type == "accuracy":
        test_inference_accuracy(backend, config)
    else:
        raise ValueError('Unsupported test type {}'.format(config.test_type))
