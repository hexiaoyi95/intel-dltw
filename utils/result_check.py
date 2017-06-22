from utils.io import json2dict
import numpy as np
import logging
logger = logging.getLogger('root')
PRECISION = 1e-4
import pprint
import os
import math

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

def find_fail(data, data_ref, ctx, precision):
    result = list()
    count = 0
    if data.size == data_ref.size:
        
        difrens = abs(data - data_ref) - abs(data_ref)*1e-02 - precision
        for index,val in np.ndenumerate(difrens):
            if val >= 0 or math.isnan(data[index]) or math.isnan(data_ref[index]):
                count +=1
                result.append([count, index, str(data[index]) , str(data_ref[index])])
            if count >= 100:
                count = 0
                break
    else:
        raise Exception('compared arrys shape not match %d vs %d' %(data.size,data_ref.size))
    result.insert(0,['id','coordinate','test value','reference value'])
    if count == 0:
        result.insert(1, [ctx, 'blob shape: ' + str(data.shape), 'total fail: >{:.4f}%'.format(100.0/data.size*100)])
    else:
        result.insert(0,[ctx, 'blob shape: '+ str(data.shape) , 'total fail: {:.4f}%'.format(float(count)/data.size*100)])
    return result

def layer_accuracy_convergence(backend, test_result, out_dir, ref_dir, precision=1e-04 ):

    this_batch_result = list()
    this_batch_result.append(['-']*40)

    count = -1

    last_layer_name = test_result.keys()[len(test_result)-1]
    fwd_accuracy = 'pass'
    bwd_accuracy = 'pass'
    update_accuracy = 'pass'
    first_param = True
    for layer_name, l in test_result.iteritems():
        count +=1
        this_layer_pass = 'pass'
        this_layer_result = list()
        detailTXT = list()
        for j, [blob_name, np_list] in enumerate(l):
            ref_sample_list = list()
            sample_list = list()

            for i, np_arry in enumerate(np_list):
                blob_title = list()
                if blob_name == 'params_diff':
                    ctx = 'params_{}_diff'.format(i)
                elif blob_name == 'params_data':
                    ctx = 'params_{}_data'.format(i)
                else:
                    if i == 0:
                        ctx = 'blob_{}_data'.format(j)
                    else:
                        ctx = 'blob_{}_diff'.format(j)
                try:
                    ref_data = np.load(os.path.join(ref_dir,layer_name.replace('/', '-'), blob_name + '_' + ctx + '.npy'))
                except IOError:
                    logger.warn("blob {} not found in refenence, skiping ...".format(blob_name))
                    this_layer_result.append(['can not find {} in reference,skiped'.format(ctx)])
                    continue

                isequal = np.allclose(np_arry, ref_data,  rtol=1e-02, atol=precision, equal_nan = True)

                if isequal:
                    this_arry = 'pass'
                    blob_title.append(ctx + ': ' + this_arry)
                    this_layer_result.append(blob_title)
                else:
                    this_arry = 'fail'
                    this_layer_pass = 'fail'
		    #logger.debug('layer_name {}, blob_name {}, {} '.format(layer_name, blob_name, i))
                    datail_diff = find_fail(np_arry, ref_data, ctx , precision)
                    this_layer_result.extend(datail_diff[:11])
                    detailTXT.extend(datail_diff)

                    if layer_name == last_layer_name and ctx == 'data':
                        fwd_accuracy = 'fail'

                    if (blob_name == 'params_diff' or ctx == 'diff')and first_param:
                        bwd_accuracy = 'fail'
                        first_param = False

                    if blob_name == 'params_data':
                        update_accuracy = 'fail'


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
            with open(os.path.join(out_dir, layer_name.replace('/','-'), 'fail detail.txt'),'w') as fp:
                for line in detailTXT:
                    for word in line:
                        fp.write(str(word))
                        if type(word) == type('?') and  word == '-' :
                            continue
                        fp.write('\t')
                    fp.write('\n')
                fp.write('\n')
                fp.write('\n')

    this_batch_result.insert(0,['weight update: ', update_accuracy])
    this_batch_result.insert(0,['net backward: ', bwd_accuracy])
    this_batch_result.insert(0,['net forward: ', fwd_accuracy])

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

                isequal = np.allclose(np_arry, ref_data,  rtol=1e-02, atol=precision, equal_nan = True)

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


