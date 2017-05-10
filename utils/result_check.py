from utils.io import json2dict
import numpy as np
import logging
logger = logging.getLogger()
PRECISION = 1e-4
import pprint
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


def check_layer_accuracy_result(batch_name, test_datas, test_weights,ref_dir, check_result):
    last_res = check_result
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

    for key in test_datas:
        ref_data = np.load(ref_dir + '/' + num + '/' + key + '_' + 'datas' + '.npy')

        if np.average(ref_data) < 1e-06 and np.average(test_datas[key]) < 1e-06:
            data_isequal = True
        else:
            data_isequal = np.allclose(test_datas[key], ref_data,  rtol=1e-01, atol=0, equal_nan = True)

        if data_isequal == False:
            logger.debug(key)
            #print abs(test_datas[key] - ref_data)
            #print test_datas[key]
            #print test_datas[key] - ref_data
        if first_result:
            last_res[key] = data_isequal
        else:
            last_res[key] &= data_isequal

        for key in test_weights:

            ref_weight = np.load(ref_dir + '/' + num + '/' + key + '_' + 'weights' + '.npy')
            if np.average(ref_weight) < 1e-06 and np.average(test_weights[key]) < 1e-06:
                weight_isequal = True
            else:
                weight_isequal= np.allclose(test_weights[key], ref_weight, rtol=1e-01, atol=0, equal_nan = True)


            
           
          


            if weight_isequal == False:
                logger.debug(key)
                #print test_weights[key] - ref_weight

            if first_result:
                last_res[key] = weight_isequal
            else:
                last_res[key] &= weight_isequal



    return last_res



