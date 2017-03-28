import sys, os
import logging
from utils.benchmark import get_elapsed_ms, performance_analysis, shuffle_input
from backends import backends_factory

logger = logging.getLogger('root')

def test_inference_performance(config):
    logger.debug("testing performance for image classification")
    backend_class = backends_factory(config.backend)
    backend = backend_class()

    backend.prepare_classify(config)


    elpased_ms_list = []

    for it in xrange(config.iteration):
        backend.shuffle_inputs(backend.inputs)
        elapsed_ms = get_elapsed_ms(backend.infer)
        elpased_ms_list.append(elapsed_ms)

    performance_analysis(elpased_ms_list)

def test_inference_accuracy(config):
    logging.debug("testing performance for image classification")
    return

def run(config):
    if config.test_type == "performance":
        test_inference_performance(config)
    elif config.test_type == "accuracy":
        test_inference_accuracy(config)
    else:
        raise ValueError('Unsupported test type {}'.format(config.test_type))
