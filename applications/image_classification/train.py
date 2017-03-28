import sys, os
import logging
from utils.benchmark import get_elapsed_ms, performance_analysis, shuffle_input
from backends import backends_factory

def test_train_performance(config):
    print "testing performance for training"
    return

def test_train_accuracy(config):
    print "testing accuracy for training"
    return

def run(config):
    if config.test_type == "performance":
        test_train_performance
    elif config.test_type == "accuracy":
        test_train_accuracy
    else:
        raise ValueError('Not supported test type {}'.format(config.test_type))
