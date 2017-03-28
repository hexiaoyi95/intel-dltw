import sys, os
import logging
from utils.benchmark import get_elapsed_ms, performance_analysis, shuffle_input
from backends import backends_factory

def test_inference_performance(config):
    print "testing performance for image classification"
    return
    backend_class = backends_factory(config.backend)
    backend = backend_class()

    backend.prepare_infer(config)

    elpased_ms_list = []

    for it in xrange(config.iteration):
        shuffle_input(backend.input)
        backend.set_input()
        elapsed_ms = get_elapsed_ms(backend.infer)
        elpased_ms_list.append(elapsed_ms)

    performance_analysis(elpased_ms_list)

def test_inference_accuracy(config):
    print "testing performance for image classification"
    return

def run(config):
    if config.test_type == "performance":
        test_inference_performance
    elif config.test_type == "accuracy":
        test_inference_accuracy
    else:
        raise ValueError('Not supported test type {}'.format(config.test_type))
