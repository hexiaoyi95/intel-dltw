#!/usr/bin/env python
# encoding: utf-8
import sys, os
import logging
import utils
from utils.benchmark import Timer
from backends import backends_factory
import pprint
logger = logging.getLogger()

def get_layer_perf(layer_id, direction, backend, config):
    if direction == "forward":
        go_through = backend.forward_layer
    elif direction == "backward":
        go_through = backend.backward_layer
    else:
        raise Exception('Expect forward or backward but get {}'.format(direction))

    backend.prepare_benchmark(config)

    elapsed_ms_list = []
    timer = Timer()
    for i in xrange(int(config.iteration)):
        timer.start()
        go_through(layer_id)
        timer.stop()
        elapsed_ms_list.append(timer.milliseconds())

    avg_time = utils.benchmark.performance_analysis(elapsed_ms_list)[0]
    FPS = config.batch_size / (avg_time / 1000.0)
    return [backend.get_layer_name(layer_id), backend.get_layer_type(layer_id), avg_time, FPS]

def get_layers_perf(direction, backend, config):
    """
    return
        per layer forward or backward time: [(layer name, layer type, elapsed_time, FPS), ... ]
    """
    total_time = 0.0
    layers_perf = []
    for layer_id in xrange(len(backend.layers())):
        layer_perf = get_layer_perf(layer_id, direction, backend, config)
        total_time += layer_perf[2]
        layers_perf.append(layer_perf)

    FPS = config.batch_size / (total_time / 1000.0)
    layers_perf.append(['total', 'summary ', total_time, FPS])
    return layers_perf

def get_net_perf(direction, backend, config):
    """
    return
        net forward or backward time: float
    """
    if direction == "forward":
        go_through = backend.forward
    elif direction == "backward":
        go_through = backend.backward
    else:
        raise Exception('Expect forward or backward but get {}'.format(direction))
    elapsed_ms_list = []
    timer = Timer()
    for i in xrange(5):
        backend.forward()
        backend.backward()
    for i in xrange(int(config.iteration)):
        timer.start()
        go_through()
        timer.stop()
        elapsed_ms_list.append(timer.milliseconds())

    avg_time = utils.benchmark.performance_analysis(elapsed_ms_list)[0]
    FPS = config.batch_size / (avg_time / 1000.0)
    return avg_time, FPS

def run(config):
    """
    return:
        Per layer forward performance: [(layer name, layer type, elapsed_time, FPS), ... ]
        Per layer backward performance:[(layer name, layer type, elapsed_time, FPS), ... ]
    """
    backend_class = backends_factory(config.backend)
    backend = backend_class(config)
    layers_forward_perf = get_layers_perf('forward', backend, config)
    layers_backward_perf = get_layers_perf('backward', backend, config)
    net_forward_perf = get_net_perf('forward', backend, config)
    net_backward_perf = get_net_perf('backward', backend, config)
    res_dict = {
        'layers_forward_perf' : layers_forward_perf,
        'layers_backward_perf': layers_backward_perf,
        'net_forward_perf'    : net_forward_perf,
        'net_backward_perf'   : net_backward_perf
    }

    logger.debug(pprint.pformat(layers_forward_perf[-1]))
    logger.debug(pprint.pformat(layers_backward_perf))
    logger.debug(pprint.pformat(net_forward_perf))
    logger.debug(pprint.pformat(net_backward_perf))

    #write res_dict to file
    out_dir = os.path.expanduser(str(config.out_dir))
    out_path = os.path.join(out_dir, "perf_data.json")
    utils.io.dict2json(res_dict, out_path)

