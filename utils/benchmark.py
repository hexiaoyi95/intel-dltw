#!/usr/bin/env python
# encoding: utf-8
import time
import numpy

def inference_performance(iteration, shuffle_input, target_method):
    """
    obj contain:
        data: input data
        method: shuffle_input
        method: target_method
    return: list
    """
    #Avg time
    elpased_times = []
    for i in xrange(iteration):
        obj.shuffle_input()

        ts = time.time()
        obj.method()
        te = time.time()

        elpased_times.append( (te -ts) * 1000 )

    return elpased_times

def performance_analysis(perf_list):
    """
    """
    if len(perf_list) > 1:
        perf_list.pop(0)
    mean = numpy.mean(perf_list)
    std = numpy.std(perf_list)
    print perf_list
    print mean, std
    return mean, std



def get_elapsed_ms(method):
    ts = time.time()
    method()
    te = time.time()
    return (te-ts)*1000


def shuffle_input():
    pass
