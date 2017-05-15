#!/usr/bin/env python
# encoding: utf-8
import sys, os
import logging
import utils
from utils.benchmark import Timer
from backends import backends_factory
import pprint
from utils.io import json2dict
logger = logging.getLogger()

def get_layers_perf(direction, backend, config):
    """
    return
        per layer forward or backward time: [(layer name, layer type, elapsed_time, FPS), ... ]
    """
    layers_perf = []
    elapsed_ms_list = []
    for i in xrange(int(config.iteration)):
        backend.prepare_benchmark(config)
        [layers_time,net_time]=backend.get_layers_perf(direction)

        elapsed_s_layers_list = [[] for l in xrange(len(layers_time))]
        elapsed_ms_list.append(net_time)

        for j in xrange(len(layers_time)):
            elapsed_s_layers_list[j].append(layers_time[j][1])

    for k in xrange(len(elapsed_s_layers_list)):
        func_name = layers_time[k][0]
        avg_time = utils.benchmark.performance_analysis(elapsed_s_layers_list[k])[0]
        layer_perf = [func_name, avg_time]
        layers_perf.append(layer_perf)

    net_avg_time = utils.benchmark.performance_analysis(elapsed_ms_list)[0]
    net_FPS = config.batch_size / ( net_avg_time / 1000 )
    net_perf = [net_avg_time,net_FPS]

    layers_perf.append(['summary', [net_avg_time, net_FPS]])

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
    for i in xrange(1):
        backend.prepare_benchmark(config)
        backend.forward()
        backend.backward()

    for i in xrange(int(config.iteration)):
        backend.prepare_benchmark(config)
        timer.start()
        go_through()
        timer.stop()
        elapsed_ms_list.append(timer.milliseconds())

    avg_time = utils.benchmark.performance_analysis(elapsed_ms_list)[0]
    FPS = config.batch_size / (avg_time / 1000.0)
    return [avg_time, FPS]

def convertToReport(res_dict, config):
    ref_res_dict = json2dict(os.path.join(config.reference.result_dir, 'perf_data.json'))
    aTXT = list()
    net_time = list()
    aTXT.append('net performance: ')
    net_time.append('forward: ')
    net_time.append('%.4f' % (res_dict['net_forward_perf'][0]))
    net_time.append('%.4f' % (ref_res_dict['net_forward_perf'][0]))
    net_time.append('%.2f' % (-100*(res_dict['net_forward_perf'][0] - ref_res_dict['net_forward_perf'][0])/ref_res_dict['net_forward_perf'][0]) + '%')
    aTXT.append(net_time)
    net_time = list()
    net_time.append('backward: ')
    net_time.append('%.4f' % (res_dict['net_backward_perf'][0]))
    net_time.append('%.4f' % (ref_res_dict['net_backward_perf'][0]))
    net_time.append('%.2f' %(-100*(res_dict['net_backward_perf'][0] - ref_res_dict['net_backward_perf'][0])/ref_res_dict['net_backward_perf'][0]) + '%')
    aTXT.append(net_time)
    aTXT.append(['-']*40)
    layers_f_perf = dict(res_dict['layers_forward_perf'])
    layers_b_perf = dict(res_dict['layers_backward_perf'])
    ref_f_perf = dict(ref_res_dict['layers_forward_perf'])
    ref_b_perf = dict(ref_res_dict['layers_backward_perf'])

    orderedKey = sorted(layers_f_perf.iterkeys(), key = lambda item:item.split('_',1)[0])
    for key in orderedKey:
        if key != 'summary':
            #print key
            layer_time = list()
            layer_time.append('layer_id: ' + key.split('_',1)[0])
            layer_time.append('layer_name: ' + key.split('_',1)[1])
            aTXT.append(layer_time)
            layer_time = list()
            layer_time.append('forward time(ms): ')
            layer_time.append('%.4f' % (layers_f_perf[key]))
            layer_time.append('%.4f' % (ref_f_perf[key]))
            layer_time.append( '%.2f' % (-100*(layers_f_perf[key] - ref_f_perf[key])/ref_f_perf[key]) + '%')
            aTXT.append(layer_time)
            layer_time = list()
            layer_time.append('backward time(ms): ')
            layer_time.append('%.4f' % (layers_b_perf[key]))
            layer_time.append('%.4f' % (ref_b_perf[key]))
            layer_time.append( '%.2f' % (-100*(layers_b_perf[key] - ref_b_perf[key])/ref_b_perf[key]) + '%')
            aTXT.append(layer_time)
            aTXT.append(['-']*40)

    with open(os.path.join(config.out_dir, 'performance_report'),'w') as fp:
        for line in aTXT:
            if type(line) == type([]):
                for word in line:
                    #print word
                    fp.write(str(word))
                    if type(word) == type('?') and  word == '-' :
                        continue
                    fp.write('\t')
            else:
                fp.write(line)
            fp.write('\n')
        fp.write('\n')
        fp.write('\n')
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

    if hasattr(config, 'getReport'):
        convertToReport(res_dict,config)

    if hasattr(config, 'reference'):
        ref_res_dict = json2dict(os.path.join(config.reference.result_dir,'perf_data.json'))

        for key,value in res_dict.iteritems():
            try:
                ref_value = ref_res_dict[key]
            except:
                raise Exception('Unexpected reference ')

            if 'layers' in key:
                ref_perf_dict = dict(ref_value)
                for index, perf_list in enumerate(value):
                    ref_time = 0.0

                    if perf_list[0] == 'summary':
                        ref_time = ref_perf_dict[perf_list[0]][0]
                        diff_time = ref_time - perf_list[1][0]
                    else:
                        ref_time = ref_perf_dict[perf_list[0]]
                        diff_time = ref_time - perf_list[1]
                        layer_name = perf_list[0]
                        this_time = perf_list[1]


                    value[index].append("faster than ref")
                    value[index].append([diff_time, '%.2f' % (100 * diff_time / ref_time)+"%"])


            else:
                ref_time = ref_value[0]
                diff_time =   ref_value[0] - value[0]
                value.append('faster than ref')
                value.append([diff_time, '%.2f' % (100 * diff_time / ref_time)+"%"])

    logger.debug(pprint.pformat(layers_forward_perf))
    logger.debug(pprint.pformat(layers_backward_perf))
    logger.debug(pprint.pformat(net_forward_perf))
    logger.debug(pprint.pformat(net_backward_perf))

    #write res_dict to file
    out_dir = os.path.expanduser(str(config.out_dir))
    if hasattr(config, 'reference'):
        out_path = os.path.join(out_dir, "perf_cmp.json")
    else:
        out_path = os.path.join(out_dir, "perf_data.json")
    utils.io.dict2json(res_dict, out_path)

