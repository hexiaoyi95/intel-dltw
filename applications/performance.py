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

def _get_layers_perf(backend, direction, elapsed_ms_net_list, elapsed_ms_layers_list):
    [layers_time, net_time]=backend.get_layers_perf(direction)
    elapsed_ms_net_list.append(net_time)
    for layer_perf in layers_time:
        layer_id = layer_perf[0]
        layer_time = layer_perf[1]
        elapsed_ms_layers_list[layer_id].append(layer_time)

def layers_perf_post_process(config, backend, elapsed_ms_layers_list, elapsed_ms_net_list):
    layers_perf = []
    #first get perf of every layer
    for k in xrange(len(elapsed_ms_layers_list)):
        layer_name = backend.get_layer_name(k)
        avg_time = utils.benchmark.performance_analysis(elapsed_ms_layers_list[k])[0]
        layer_perf = [k, avg_time]
        layers_perf.append(layer_perf)

    #then get the sum perf of every layer
    net_perf = get_avg_FPS(elapsed_ms_net_list, config.batch_size)
    layers_perf.append(['summary', net_perf])

    return layers_perf

def get_layers_perf(backend, config):
    """
    return
        per layer forward and backward time: [(layer name, layer type, elapsed_time, FPS), ... ]
    """
    fwd_elapsed_ms_net_list = []
    fwd_elapsed_ms_layers_list = [[] for l in xrange(len(backend.layers()))]
    bwd_elapsed_ms_net_list = []
    bwd_elapsed_ms_layers_list = [[] for l in xrange(len(backend.layers()))]


    for i in xrange(int(config.iteration)):
         
        logger.debug('processing {}th iteration forward-backward layer by layer'.format(i))
        _get_layers_perf(backend, 'forward', fwd_elapsed_ms_net_list, fwd_elapsed_ms_layers_list)
        _get_layers_perf(backend, 'backward', bwd_elapsed_ms_net_list, bwd_elapsed_ms_layers_list)

    layers_forward_perf = layers_perf_post_process(config, backend, fwd_elapsed_ms_layers_list, fwd_elapsed_ms_net_list)
    layers_backward_perf =layers_perf_post_process(config, backend, bwd_elapsed_ms_layers_list, bwd_elapsed_ms_net_list)

    return layers_forward_perf, layers_backward_perf

def _get_net_perf(go_through):
    timer = Timer()
    timer.start()
    go_through()
    timer.stop()
    return timer.milliseconds()

def get_avg_FPS(elapsed_ms_list, batch_size):
    avg_time = utils.benchmark.performance_analysis(elapsed_ms_list)[0]
    FPS = batch_size / (avg_time / 1000.0)
    return [avg_time, FPS]

def get_net_perf(backend, config):

    #backend.prepare_benchmark(config)
    """
    return
        net forward and backward time: float
    """
    fwd_elapsed_ms_list = []
    bwd_elapsed_ms_list = []
    for i in xrange(int(config.iteration)):
        logger.debug('processing {}th iteration forward'.format(i))
        net_time = _get_net_perf(backend.forward)
        fwd_elapsed_ms_list.append(net_time)
        logger.debug('processing {}th iteration backward'.format(i))
        net_time = _get_net_perf(backend.backward)
        bwd_elapsed_ms_list.append(net_time)

    net_forward_perf = get_avg_FPS(fwd_elapsed_ms_list, config.batch_size)
    net_backward_perf = get_avg_FPS(bwd_elapsed_ms_list, config.batch_size)
    return net_forward_perf, net_backward_perf

def convertToReport(res_dict, config, backend):
    ref_res_dict = json2dict(os.path.join(config.reference.result_dir, 'perf_data.json'))
    aTXT = list()
    net_time = list()
    #aTXT.append("Test engine: {}, reference engine: {}".format(config.backend.engine,config.reference.engine))
    aTXT.append(['-']*80)
    aTXT.append('net performance: ')
    net_time.append('forward: ')
    net_time.append('time: {:<9.4f} ms'.format(res_dict['net_forward_perf'][0]))
    net_time.append('reference time: {:<9.4f} ms'.format(ref_res_dict['net_forward_perf'][0]))
    net_time.append('Gap: {:<6.2f}'.format(-100*(res_dict['net_forward_perf'][0] - ref_res_dict['net_forward_perf'][0])/ref_res_dict['net_forward_perf'][0]) + '%')
    aTXT.append(net_time)
    net_time = list()
    net_time.append('backward: ')
    net_time.append('time: {:<9.4f} ms'.format(res_dict['net_backward_perf'][0]))
    net_time.append('reference time: {:<9.4f} ms'.format(ref_res_dict['net_backward_perf'][0]))
    net_time.append('Gap: {:<6.2f}'.format(-100*(res_dict['net_backward_perf'][0] - ref_res_dict['net_backward_perf'][0])/ref_res_dict['net_backward_perf'][0]) + '%')
    aTXT.append(net_time)
    aTXT.append(['-']*80)
    layers_f_perf = dict(res_dict['layers_forward_perf'])
    layers_b_perf = dict(res_dict['layers_backward_perf'])
    ref_f_perf = dict(ref_res_dict['layers_forward_perf'])
    ref_b_perf = dict(ref_res_dict['layers_backward_perf'])

    fwd_perf_perctg = dict()
    bwd_perf_perctg = dict()
    for key in layers_f_perf.iterkeys():
	if key != 'summary':
 	    fwd_perf_perctg[key] = -100*(layers_f_perf[key] - ref_f_perf[key])/ref_f_perf[key]
	    bwd_perf_perctg[key] = -100*(layers_b_perf[key] - ref_b_perf[key])/ref_b_perf[key]
    if config.getReport.reportOrder == 'default':
    	orderedKey = sorted(layers_f_perf.iterkeys(), key = lambda item:item)
    elif config.getReport.reportOrder == 'forward performance':
	orderedKey = sorted(fwd_perf_perctg.iterkeys(), key = lambda item:fwd_perf_perctg[item])
    elif config.getReport.reportOrder == 'backward performance':
	orderedKey = sorted(bwd_perf_perctg.iterkeys(), key = lambda item:bwd_perf_perctg[item])
    else:
	raise Exception('Unsupported reprort order,choose default,forward performance or backward performance')
    layer_id = -1
    aTXT.append('layer by layer performance')
    aTXT.append(['-']*80)
    for key in orderedKey:
        if key != 'summary':
            #print key
            layer_id += 1
            layer_time = list()
            layer_time.append('layer_id: {}'.format(key) )
            layer_time.append('layer_name: {}'.format(backend.get_layer_name(key)))
            layer_time.append('layer_type: {}'.format(backend.get_layer_type(key)))
            aTXT.append(layer_time)
            layer_time = list()
            layer_time.append('forward: ')
            layer_time.append('time: {:<9.4f} ms'.format(layers_f_perf[key]))
            layer_time.append('reference time: {:<9.4f} ms'.format(ref_f_perf[key]))
            layer_time.append('Gap: {:<6.2f}'.format(fwd_perf_perctg[key] ) + '%')
            aTXT.append(layer_time)
            layer_time = list()
            layer_time.append('backward: ')
            layer_time.append('time: {:<9.4f} ms'.format(layers_b_perf[key]))
            layer_time.append('reference time: {:<9.4f} ms'.format(ref_b_perf[key]))
            layer_time.append('Gap: {:<6.2f}'.format( bwd_perf_perctg[key] )+ '%')
            aTXT.append(layer_time)
            aTXT.append(['-']*80)
    if not os.path.exists(config.out_dir):
        os.mkdir(config.out_dir)
    with open(os.path.join(config.out_dir,'test_report.txt'), 'w' ) as fp:
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
    #backend.prepare_benchmark(config)

    layers_forward_perf, layers_backward_perf = get_layers_perf(backend, config)
    net_forward_perf, net_backward_perf = get_net_perf(backend, config)


    res_dict = {
            'layers_forward_perf' : layers_forward_perf,
            'layers_backward_perf': layers_backward_perf,
            'net_forward_perf'    : net_forward_perf,
            'net_backward_perf'   : net_backward_perf
    }
    if hasattr(config, 'getReport') and hasattr(config, 'reference'):
        logger.debug('gen report')
        convertToReport(res_dict, config, backend)

    if hasattr(config, 'reference'):
        logger.debug('comparing')
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

