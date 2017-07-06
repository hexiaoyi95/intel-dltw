#!/usr/bin/env python
# encoding: utf-8
import os, sys
import logging
import glob
import numpy as np
import skimage.io
import json
import itertools
import copy
logger = logging.getLogger()
def get_input_list(input_path, img_num = None):
    """
    return image list
    """
    input_list = []

    input_path = os.path.expanduser(input_path)

    if os.path.isdir(input_path):
        logging.debug("Reading folder: %s" % input_path)
        input_list = [ img_path for img_path in glob.glob(input_path + '/*') ]
        if len(input_list) == 0:
            logging.error("Error no images found in folder {}".format(input_path))
            return 1

        if img_num == None:
            img_num = len(input_list)

        while len(input_list) < img_num:
            input_list += input_list

        input_list = input_list[:img_num]
        logger.debug("Got {0:d} images".format(len(input_list)))
    else:
        if os.path.exists(input_path):
            logging.debug("Got one image: %s" % input_path)
            input_list.append(input_path)
        else:
            raise Exception("Invalid input path")
    return input_list

def get_input_from_txt(input_txt, img_num = None):

    input_list = []
    input_txt= os.path.expanduser(input_txt)
    txt_path,txt_name = os.path.split(input_txt)
    file = open(input_txt,"r")
    for line in file:
        line = line[:-1] #dele \n
        input_list.append(line)

    if img_num == None:
        img_num = len(input_list)

    while len(input_list) < img_num:
        input_list += input_list

    logger.debug("Got {0:d} images".format(len(input_list)))

    return input_list

def slice2batches(input_list, batch_size):
    while len(input_list) % batch_size:
        input_list.append(input_list[-1])
    return [input_list[0+i: batch_size+i] for i in xrange(0, len(input_list), batch_size)]


def load_image(filename, color=True):
    """
    Load an image converting from grayscale or alpha as needed.

    Parameters
    ----------
    filename : string
    color : boolean
        flag for color format. True (default) loads as RGB while False
        loads as intensity (if image is already grayscale).

    Returns
    -------
    image : an image with type np.float32 in range [0, 1]
        of size (H x W x 3) in RGB or
        of size (H x W x 1) in grayscale.
    """
    img = skimage.img_as_float(skimage.io.imread(filename, as_grey=not color)).astype(np.float32)
    if img.ndim == 2:
        img = img[:, :, np.newaxis]
        if color:
            img = np.tile(img, (1, 1, 3))
    elif img.shape[2] == 4:
        img = img[:, :, :3]
    return img


from collections import namedtuple
def json2obj(json_path):
    json_path = os.path.expanduser(json_path)
    with open(json_path, 'r') as fp:
        data = fp.read().replace('\n', '')
    obj = json.loads(data, object_hook=lambda d: namedtuple('X', d.keys())(*d.values()))
    return obj

def json2dict(json_path):
    json_path = os.path.expanduser(json_path)
    with open(json_path, 'r') as fp:
        return json.load(fp)

def dict2json(d, json_path):
    json_path = os.path.expanduser(json_path)
    if not os.path.exists(os.path.dirname(json_path)):
        os.makedirs(os.path.dirname(json_path))
    with open(json_path, 'w') as fp:
        json.dump(d, fp, indent=4)

def genConfFilename(json_path, getJson_only= True):
    
    confDict =json2dict(json_path)
    app = confDict['application']
    ref = confDict['ref']
    ref_confList = confDict[ref]
    ref_confValue = ref_confList['ref']
    template_init = json2dict( 'test-config/' + app + '-template.json')
    iter_list = list()
    ref_list = list()
    for i, arg in confDict[ref].iteritems():
        if i != 'ref':
            ref_list.append([ref, arg])
        else: 
            ref_list.insert(0,[ref, arg])
    iter_list.append(ref_list)

    for item, argDict in confDict.iteritems():
        if item != 'application' and item != 'ref' and item !=ref :
            l = list()
            for i, arg in argDict.iteritems():
                l.append([item,arg])
            iter_list.append(l)
    
    title_generated = False
    result_list = list()
    with open('result.txt','w') as fp:
        for confList in itertools.product(*iter_list):
            #print ref_conf,ref_confValue, confList
            ref_dir = app
            out_dir = app 
            is_ref = False
            template = copy.deepcopy(template_init)
            if not title_generated:
                for [confName,_] in confList:
                    fp.write(confName + '\t')
                fp.write('report')
                fp.write('\n')
                title_generated = True
            for [confName, value] in confList:
                template[confName] = value
                out_dir += '_' + value
                fp.write(value + '\t')
                if confName == ref:
                    ref_dir += '_' + ref_confValue
                    if ref_confValue == value:
                        is_ref = True
                else: 
                    ref_dir += '_' + value
            if not is_ref:
                template['reference_dir'] = 'out/' + ref_dir 
            fp.write(out_dir + '/test_report.txt')
            fp.write('\n')
            template['out_dir'] = 'out/' + out_dir
            jsonPath = 'test-config-debug/' + out_dir + '.json'
            if getJson_only:
                result_list.append(jsonPath)
            else:
                result_list.append( [jsonPath, template])
    return result_list
        
