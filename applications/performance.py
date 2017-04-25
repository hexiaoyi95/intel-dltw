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
    layer_ids = range(len(backend.layers()))
    if direction == 'backward':
	layer_ids = range(len(backend.layers), -1, -1)

    for layer_id in layer_ids:
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

def get_chainer_perf(backend, config):
    """
    return
          net forward and backward time
    """
    backend.prepare_benchmark(config)
    layers_perf_for_back = []
    net_perf_for_back = []

    elapsed_s_list = []
    for i in xrange(1):
        backend.prepare_benchmark(config)
        backend.forward()
        backend.backward()

    [layers_f_time,net_f_time]=backend.get_net_forward_perf()
    [layers_b_time,net_b_time]=backend.get_net_backward_perf()
    layers_time = [layers_f_time, layers_b_time]

    # if len(layers_f_time) != len(layers_b_time):
    #     raise Exception('the number of forward layers is not equal backward layers')


    for index in range(2):
        layers_perf = []
        for i in xrange(int(config.iteration)):

            if index == 0:
                [layers_time,net_time]=backend.get_net_forward_perf()
            else:
                [layers_time,net_time]=backend.get_net_backward_perf()

            elapsed_s_layers_list = [[] for l in xrange(len(layers_time))]
            elapsed_s_list.append(net_time)

            for j in xrange(len(layers_time)):
                elapsed_s_layers_list[j].append(layers_time[j][1])

        for k in xrange(len(elapsed_s_layers_list)):
            avg_time = utils.benchmark.performance_analysis(elapsed_s_layers_list[k])[0]
            FPS = config.batch_size / avg_time
            layer_perf = [k, avg_time, FPS]
            layers_perf.append(layer_perf)

        net_avg_time = utils.benchmark.performance_analysis(elapsed_s_list)[0]
        net_FPS = config.batch_size / net_avg_time
        net_perf = [avg_time,net_FPS]

        layers_perf_for_back.append(layers_perf)
        net_perf_for_back.append(net_perf)

        logger.debug("Done for %d/%d" % (i, config.iteration))

    return layers_perf_for_back,net_perf_for_back


def run(config):
    """
    return:
        Per layer forward performance: [(layer name, layer type, elapsed_time, FPS), ... ]
        Per layer backward performance:[(layer name, layer type, elapsed_time, FPS), ... ]
    """
    backend_class = backends_factory(config.backend)
    backend = backend_class(config)
    if config.backend.class_path.rsplit('.', 1)[1] != "ChainerBackend":
        layers_forward_perf = get_layers_perf('forward', backend, config)
        layers_backward_perf = get_layers_perf('backward', backend, config)
        net_forward_perf = get_net_perf('forward', backend, config)
        net_backward_perf = get_net_perf('backward', backend, config)

    else:
        perf = get_chainer_perf(backend, config)
        layers_forward_perf = perf[0][0]
        layers_backward_perf = perf[0][1]
        net_forward_perf = perf[1][0]
        net_backward_perf = perf[1][1]

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

