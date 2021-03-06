from utils.io import json2dict
import numpy as np
import logging
logger = logging.getLogger('root')
PRECISION = 1e-4
import pprint
import os
import math
from utils.benchmark import Timer
from utils.io import dict2json
import google.protobuf.text_format as txtf

def iscloseFP(a, b, precision=PRECISION):
    return abs(a-b) < precision

def IOU(det_1, det_2):

    xmin_1 = det_1[0]
    ymin_1 = det_1[1]
    xmax_1 = det_1[2]
    ymax_1 = det_2[3]

    xmin_2 = det_2[0]
    ymin_2 = det_2[1]
    xmax_2 = det_2[2]
    ymax_2 = det_2[3]

    intersec_x = (xmax_1 if xmax_1 < xmax_2 else xmax_2) \
        - (xmin_1 if xmin_1 > xmin_2 else xmin_2)
    intersec_y = (ymax_1 if ymax_1 < ymax_2 else ymax_2)\
        - (ymin_1 if ymin_1 > ymin_2 else ymin_2)

    intersec_x = 0 if intersec_x < 0 else intersec_x
    intersec_y = 0 if intersec_y < 0 else intersec_y

    union_size = (xmax_1 - xmin_1) * (ymax_1 - ymin_1)\
        + (xmax_2 - xmin_2) * (ymax_2 - ymin_2)\
        - (intersec_x * intersec_y)

    return float(intersec_x * intersec_y) / union_size

def isEqualPrediction(pred, ref_pred):
    """
    Parameters:
        ref_pred: predictions for an img: [[label, prob], [lable, prob] ...]
        pred:     [[label, prob], [lable, prob] ...]

    Return:
    """
    res = [True, True]
    results = []
    for p, rp in zip(pred, ref_pred):
        #label = p[0], prob = p[1]
        if p[0] != rp[0]:
            res[0] = False

        if not np.allclose(p[1], rp[1],  rtol=1e-03, atol=1e-0, equal_nan = True):
            res[1] = False

        results.append(res[:])

    return results

def isEqualDetection(detec, ref_detec_):
    """
    Parameters:
        ref_detec: detections for an img [[conf,label,xmin,ymin,xmax,ymax]...]
        detec:
    Return:[ nums of bboxes isequal,
             conf of bboxes isclose,
             labels of bboxes isequal,
             bboxes pos isequal
             ]
    """
    ref_detec = ref_detec_[:]
    res = [True, True, True,True]

    if(len(detec) != len(ref_detec)):
        res[0] = False

    for i in range(len(detec)):
        iou = []
        for j in range(len(ref_detec)):
            iou.append(IOU(detec[i][2:], ref_detec[j][2:]))
        if len(iou) > 0:
            max_iou = max(iou)
            if max_iou < 0.5:
                continue
        else:
            continue
        #pprint.pprint(iou)
        index = iou.index(max_iou)
        #print i,index
        if not iscloseFP(detec[i][0], ref_detec[index][0]):
            res[1] = False

        if detec[i][1] != ref_detec[index][1]:
            res[2] = False
            print detec[i][1],ref_detec[index][1]

        if iou < 0.99:
            res[3] = False

        del ref_detec[index]

    return res


def update_result(results, new_results):
    for i in xrange(len(results)):
        for j in xrange(len(results[i])):
            results[i][j] &= new_results[i][j]

def update_result_obj(results, new_results):
    changed = False
    for i in xrange(len(results)):
        if results[i] == True and new_results[i] == False:
            changed = True

        results[i] &= new_results[i]
    return changed

def res_as_dict(results):
    res_dict = {}
    for i, item in enumerate(results):
        res_dict['top_{}'.format(i+1)] = item
    return res_dict

def res_as_dict_for_obj(results):
    res_dict = {}

    res_dict['nums of bboxes is euqal'] = results[0]
    res_dict['confs of bboxes is close'] = results[1]
    res_dict['labels of bboxes is the same'] = results[2]
    res_dict['pos of bboxes is close'] = results[3]
    return res_dict

def check_classify_result(test_file, reference_file):
    """
    return: a list with index N, list[N] is the result of topN, include label check and accuracy check.
            [[True True], [True, True], [True, False], ....]
            The result of topN means the result of first N predictions
    """
    test_data = json2dict(test_file)
    ref_data = json2dict(reference_file)
    results = None
    for img in test_data:
        if not img in ref_data:
            raise Exception('image can not be found in reference data')
        else:
            new_results = isEqualPrediction(test_data[img], ref_data[img])
            if not results:
                results = new_results
            else:
                update_result(results, new_results)

    return res_as_dict(results)

def check_detection_result(test_file,reference_file,):

    """
    Return:[ [nums of bboxes isequal,
             conf of bboxes isclose,
             labels of bboxes isequal,
             bboxes pos isequal]
             ....
             [nums of bboxes isequal,
             conf of bboxes isclose,
             labels of bboxes isequal,
             bboxes pos isequal]
            ]

    """

    test_data = json2dict(test_file)
    ref_data = json2dict(reference_file)
    results = None
    for img in test_data:
        if not img in ref_data:
            raise Exception('image can not be found in reference data')
        else:
            new_results = isEqualDetection(test_data[img], ref_data[img])
            if not results:
                results = new_results
            else:
                changed = update_result_obj(results, new_results)
                if changed:
                    print img

    return res_as_dict_for_obj(results)


def check_layer_accuracy_result(batch_name, test_datas, test_diffs,ref_dir, check_result):
    last_res = check_result
    flag = 0
    if len(last_res) == 0:
        first_result = True
    else:
        first_result = False

    ref_json = ref_dir + '/' + 'name.json'
    ref_batches_name = json2dict(ref_json)

    for num, img_list in batch_name.iteritems():
        if not num in ref_batches_name :
            raise Exception('batch can not be found in reference data')

        for img in img_list:
            if not img in ref_batches_name[num]:
                raise Exception('image in batch %s can not be found in reference data ' % (num))
    ordered_key = sorted(test_datas.keys(),key = lambda item:item.split('_')[0])
    for key in ordered_key:
        ref_data = np.load(ref_dir + '/' + num + '/' + key.replace('/', '-') + '_' + 'datas' + '.npy')

        data_isequal = np.allclose(test_datas[key], ref_data,  rtol=1e-02, atol=1e-04, equal_nan = False)

        if first_result:
            last_res[key] = data_isequal
        else:
            if data_isequal == False and flag == 0:
                logger.debug('error occur from  '  + key)
                last_res['first_error'] = key
                #print key
                #print test_datas[key] - ref_data
                flag = 1
            last_res[key] &= data_isequal

    ordered_key = sorted(test_diffs.keys(),key = lambda item:item.split('_')[0], reverse =True)

    for key in ordered_key:
        ref_weight = np.load(ref_dir + '/' + num + '/' + key + '_' + 'diffs' + '.npy')
        if np.average(ref_weight) < 1e-06 and np.average(test_diffs[key]) < 1e-06:
            weight_isequal = True
        else:
            weight_isequal= np.allclose(test_diffs[key], ref_weight, rtol=1e-02, atol=1e-04, equal_nan = False)

        if first_result:
            last_res[key] = weight_isequal
        else:
            if weight_isequal == False and flag== 0:
                logger.debug('error occur from  ' + key)
                last_res['first_error'] = key
                #print key
                #print test_diffs[key] - ref_weight
                flag = 1
            last_res[key] &= weight_isequal

    return last_res


def check_error(data, data_ref, ctx, rtol, atol, check_method):
    result = list()
    count = 0
    check_result = True
    if data.shape == data_ref.shape:
        if check_method == 'mklUnitTest':
            with np.errstate(invalid='ignore'):
                error=np.where( abs(data_ref) <= atol, data-data_ref, (data-data_ref)/data_ref )
            pick_array = rtol < abs(error)
            isequal = not pick_array.any()
            if isequal:
                return isequal,result
        elif check_method == 'npAllClose':
            isequal = np.allclose(data, data_ref, rtol=rtol, atol=atol, equal_nan=False)
            if isequal:
                return isequal,result
            pick_array = np.less_equal(abs(data_ref)*rtol  + atol, abs(data - data_ref))
        else:
            raise Exception("unExcepted check method")
        data_fail = data[pick_array]   
        data_ref_fail = data_ref[pick_array]
        index_fail = np.transpose(np.nonzero(pick_array))
        count = len(data_fail)
        #only get 100 failed value at most
        fail_num = len(data_fail) if len(data_fail) < 100 else 100
        for i in range(fail_num):
            result.append([str(index_fail[i]), str(data_fail[i]) , str(data_ref_fail[i])])

    else:
        logger.warn('compared arrys shape not match %s vs %s'  \
            %(str(data.shape),str(data_ref.shape)))
        return False,[[ctx,"the shape of test data: %s do not match the one of reference: %s" \
            % (str(data.shape),str(data_ref.shape))]]

    result.insert(0,['index','test value','reference value'])
    
    result.insert(0,[ctx, 'blob shape: '+ str(data.shape) , \
            'total fail: {}/{}'.format(count,data.size)])

    return isequal,result

def layer_accuracy_convergence(backend, test_result, out_dir, ref_dir, config, rtol=1e-02, atol=1e-04, check_method='npAllClose'):

    this_batch_result = list()
    this_batch_result.append(['-']*40)

    count = -1
    accuracy_level = ''
    if hasattr(config,'forward_only') and not config.forward_only and \
        config.model.prototxt_type == 'train_val':
        accuracy_level = 'bwd'
    last_layer_name = test_result.keys()[len(test_result)-1]
    fwd_accuracy = 'pass'
    bwd_accuracy = 'pass'
    update_accuracy = 'pass'
    test_result_str = 'pass'
    first_param = True
    first_blob_fail = True
    for layer_name, l in test_result.iteritems():
        count +=1
        this_layer_pass = 'pass'
        this_layer_result = list()
        detailTXT = list()
        for j, [blob_name, np_list] in enumerate(l):
            #blob_name = real_blob_name + '_data' or '_diff'
            ref_sample_list = list()
            sample_list = list()

            for i, np_arry in enumerate(np_list):
                blob_title = list()
                if blob_name == 'params_diff':
                    ctx = layer_name + '_params_{}_diff'.format(i)
                elif blob_name == 'params_data':
                    ctx = layer_name + '_params_{}_data'.format(i)
                else:
                    ctx = blob_name
                try:
                    ref_data = np.load(os.path.join(ref_dir , ctx.replace('/','-') + '.npy'))
                except IOError:
                    logger.warn("blob {} not found in refenence, skiping ...".format(blob_name))
                    this_layer_result.append(['can not find {} in reference,skiped'.format(ctx)])
                    continue
                #deal with the case in which data's size match but shape not match
                if np_arry.size == ref_data.size and np_arry.shape != ref_data.shape:
                    np_arry = np_arry.reshape( ref_data.shape)    

                if is_data_layer(backend.get_layer_type(backend.get_layer_id(layer_name))):
                    rtol_real= 0.0
                    atol_real= 0.0
                else:
                    rtol_real = rtol
                    atol_real = atol

                isequal,detail_diff=check_error(np_arry,ref_data, ctx, \
                                        rtol_real, atol_real, check_method)

                if isequal:
                    this_arry = 'pass'
                    blob_title.append(ctx + ': ' + this_arry)
                    this_layer_result.append(blob_title)
                else:
                    this_arry = 'fail'
                    this_layer_pass = 'fail'
                    #detail_diff = find_fail(np_arry, ref_data, ctx , precision)
                    this_layer_result.extend(detail_diff[:11])
                    detailTXT.extend(detail_diff)
                    if ctx == blob_name and first_blob_fail:
                        if hasattr(config, 'reproduce'):
                            generate_data_for_reproduce(backend, blob_name, layer_name, \
                                    ref_dir, out_dir, config)
                        first_blob_fail = False
                    if accuracy_level == 'bwd':
                        if blob_name == 'params_diff':
                            test_result_str = 'fail'
                    else:
                        test_result_str = 'fail'
                   # if ctx == blob_name +'_data' and layer_name == last_layer_name:
                   #     fwd_accuracy = 'fail'

                   # if (blob_name ==  'params_diff' or ctx == blob_name + '_diff'):
                   #     bwd_accuracy = 'fail'
                   # 
                   # if blob_name == 'params_data':
                   #     test_result_str = 'fail'
        

            #     ref_sample_list.append(ctx + '_ref' + ': ')
            #     ref_sample_list.append(np.concatenate((ref_data.flatten()[1:6],ref_data.flatten()[-5:])))
            #     sample_list.append(ctx + ': ')
            #     sample_list.append(np.concatenate((np_arry.flatten()[1:6],np_arry.flatten()[-5:])))

            # sample_list.insert(0, '    ')
            # ref_sample_list.insert(0, '    ')
            #this_layer_result.append(sample_list)
            #this_layer_result.append(ref_sample_list)

        # add the tile to the result of this layer
        this_layer_result.insert(0,['layer id: %04d' % count, 'layer type: '+ backend.get_layer_type(count), 'layer name: '+ layer_name, this_layer_pass])
        detailTXT.insert(0,['%04d' % count, backend.get_layer_type(count), layer_name, this_layer_pass])
        this_layer_result.insert(0,['-']*40)
        detailTXT.insert(0,['-']*40)

        #add the result of this layer to total result
        this_batch_result.extend(this_layer_result)

        if this_layer_pass == 'fail':
            with open(os.path.join(out_dir, layer_name.replace('/','-') + '_fail_detail.txt'),'w') as fp:
                for line in detailTXT:
                    for word in line:
                        fp.write(str(word))
                        if type(word) == type('?') and  word == '-' :
                            continue
                        fp.write('\t')
                    fp.write('\n')
                fp.write('\n')
                fp.write('\n')

    this_batch_result.insert(0,['Test Result: ', test_result_str])
    #this_batch_result.insert(0,['weight update: ', update_accuracy])
    #this_batch_result.insert(0,['net backward: ', bwd_accuracy])
    #this_batch_result.insert(0,['net forward: ', fwd_accuracy])
    #this_batch_result.insert(0,['Test engine: {}, reference engine: {}'.format(config.backend.engine,config.reference.engine)]) 
    return this_batch_result


def layer_accuracy_debug(batch_num, img_names, test_result,ref_dir, precision=1e-04 ):

    this_batch_result = list()
    this_batch_result.append(['batch_num: ',batch_num])
    this_batch_result.append(['-']*40)
    ref_json = ref_dir + '/' + 'name.json'
    ref_batches_name = json2dict(ref_json)

    if not str(batch_num) in ref_batches_name :
        raise Exception('batch can not be found in reference data')

    for img in img_names:
        if not img in ref_batches_name[str(batch_num)]:
            raise Exception('image in batch %s can not be found in reference data ' % (batch_num))
    count = 0
    for layer_name, l in test_result.iteritems():
        count +=1
        this_layer_pass = 'pass'
        this_layer_result = list()
        for j, [blob_name, np_list] in enumerate(l):
            ref_sample_list = list()
            sample_list = list()
            blob_title = list()
            for i, np_arry in enumerate(np_list):

                if blob_name == 'params_diff':
                    if i == 0:
                        ctx = 'W'
                    else:
                        ctx = 'b'

                else:
                    if i == 0:
                        ctx = 'data'
                    else:
                        ctx = 'diff'
                try:
                    ref_data = np.load(os.path.join(ref_dir, 'batch_' + str(batch_num),layer_name.replace('/', '-'), blob_name + '_' + ctx + '.npy'))
                except IOError:
                    logger.error("layer {} not found in refenence, skiping ...".format(layer_name))
                    continue;

                isequal = np.allclose(np_arry, ref_data,  rtol=1e-04, atol=precision, equal_nan = True)

                if isequal:
                    this_arry = 'pass'
                else:
                    this_arry = 'fail'
                    this_layer_pass = 'fail'

                blob_title.append(ctx + ': ' + this_arry)

                ref_sample_list.append(ctx + '_ref' + ': ')
                ref_sample_list.append(np.concatenate((ref_data.flatten()[1:6],ref_data.flatten()[-5:])))
                sample_list.append(ctx + ': ')
                sample_list.append(np.concatenate((np_arry.flatten()[1:6],np_arry.flatten()[-5:])))
            if blob_name == 'params_diff':
                blob_title.insert(0, '  paramaters_diff: ')
            else:
                blob_title.insert(0, '  top_name: ' + blob_name)
            this_layer_result.append(blob_title)
            sample_list.insert(0, '    ')
            ref_sample_list.insert(0, '    ')
            this_layer_result.append(sample_list)
            this_layer_result.append(ref_sample_list)
        this_batch_result.append(['%04d' % count, layer_name, this_layer_pass])
        this_batch_result.extend(this_layer_result)
        this_batch_result.append(['-']*40)

    return this_batch_result

def is_data_layer(layer_type):
    data_layer = ['ImageData','Data','HDF5Data','WindowData','MemoryData']
    if str(layer_type) in data_layer:
        return True
    else:
        return False 

def generate_data_for_reproduce(backend, blob_name, layer_name, ref_dir, out_dir, config):

    top_blob_name = blob_name.split('_')[0]
    bottom_blob_name_list = backend.get_bottom_name(layer_name)
    data_type = blob_name.split('_')[1]#'data' or 'diff'
    last_layer_id,last_layer_name=backend.get_last_layer_from_top_name(top_blob_name)
    net_params = backend.generate_net_parameter()

    bottom_data_list = list()
    ref_bottom_data_list = list()
    
    for blob_name in bottom_blob_name_list:
        bottom_data_list.append([blob_name,os.path.join(out_dir, blob_name.replace('/','-') + \
                           '_' + data_type +  '.npy')])
        ref_bottom_data_list.append([blob_name,os.path.join(ref_dir, \
                            blob_name.replace('/','-') + '_' + data_type + '.npy')])

    top_data_path = [top_blob_name, os.path.join(out_dir, top_blob_name.replace('/','-') + \
                '_' + data_type + '.npy')]
    ref_top_data_path = [top_blob_name, os.path.join(ref_dir, top_blob_name.replace('/','-') + \
                '_' + data_type + '.npy')]

    input_layers_str = 'layer {\n   name: "input"\n    type: "Input"\n '
    shape_param = 'input_param{\n'
    for blob_name, data_path in bottom_data_list:
        data = np.load(data_path)
        data_shape = data.shape
        input_layers_str += '   top: "{}"\n'.format(blob_name)
        shape = ''
        for dim in data_shape:
            shape += 'dim:{} '.format(int(dim))
        shape_param += 'shape: { %s }\n' % (shape)
    shape_param += '}\n'    
    input_layers_str += shape_param
    input_layers_str += '}\n'

    with open(str(config.model.topology)) as fp:
        s=fp.read()
        txtf.Merge(s,net_params)
         
    start_layer_id = backend.get_layer_id(layer_name)
    target_net = 'name: "single layer for reproduce bug"\n' 
    target_net += input_layers_str 

    for layer_id in range(start_layer_id, last_layer_id + 1):
        for index in range(len(net_params.layer)):
            if net_params.layer[index].name == backend.get_layer_name(layer_id):
                target_layer_str = str(net_params.layer[index])
                target_net += 'layer {\n' + \
                                target_layer_str + \
                              '}\n'
                break
     
    target_prototxt = backend.generate_net_parameter()
    txtf.Merge(target_net, target_prototxt)
    
    #next output data 
     
    output_dict = dict()
    output_dict['ref_bottom_data_path'] = ref_bottom_data_list
    output_dict['ref_top_data_path'] = ref_top_data_path
    output_dict['bottom_data_path'] = bottom_data_list
    output_dict['top_data_path'] = top_data_path
    output_dict['precision'] = {'rtol' : config.precision.rtol, 'atol' : config.precision.atol, \
                                'check_method' : config.precision.check_method}
    output_dict['weight'] = config.model.weight 
    output_dict['python_path'] = config.backend.python_path 
    output_dict['phase'] = config.model.type
    output_dict['engine'] = config.backend.engine
    output_dict['ref_config'] = config.reference.config_path
    
    save_dir = os.path.join(out_dir, 'for_reproduce')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(os.path.join(save_dir, 'test.topology'), 'w') as fp:
        fp.write(target_net)

    dict2json(output_dict, os.path.join(save_dir, 'config.json'))
             
