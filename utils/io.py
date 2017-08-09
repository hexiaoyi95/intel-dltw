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
    ref_item = confDict['ref'][0]
    ref_index = confDict['ref'][1]
    ref_value = confDict[ref_item][ref_index] 
    template_init = json2dict( 'test-config/templates-for-gen-cases/' + app + '-template.json')
    iter_list = list()
    ref_list = list()

    #insert the ref item to the head of iter list
    #each iter list include every possible value of ref item
    for i, arg in enumerate(confDict[ref_item]):
        if i != ref_index:
            ref_list.append([ref_item, arg])
        else: 
            ref_list.insert(0,[ref_item, arg])
    iter_list.append(ref_list)
    #get the rest iter list
    for item, value_list in confDict.iteritems():
        if item != 'application' and item != 'ref' and item !=ref_item :
            item_value_iter = list()
            for value in value_list:
                item_value_iter.append([item,value])
            iter_list.append(item_value_iter)
    
    #get permutaions from all iter list
    item_permutation = itertools.product(*iter_list)

    #pre-process
    modified_confs = list()
    for itemList in item_permutation:
        itemDict = conf_pre_process(dict(itemList))
        if len(modified_confs) == 0:
            modified_confs.append(itemDict)
            continue
        #remove the same config
        for conf in modified_confs:
            if len(conf.keys()) != len(itemDict.keys()) \
                or len(set(conf.items()) & set(itemDict.items())) != len(conf.keys()):
                modified_confs.append(itemDict)
                break
    
    title_generated = False
    result_list = list()
    #write the txt which including the detail of each config and the path of test report.
    cases_info =  os.path.join('test-config-debug',\
        os.path.splitext(os.path.basename(json_path))[0] + '.txt')
    cases_info_json =  os.path.join('test-config-debug',\
        os.path.splitext(os.path.basename(json_path))[0] + '_cases_info.json')
    if not os.path.exists(os.path.dirname(cases_info)):
       os.makedirs(os.path.dirname(cases_info))
    
    with open(cases_info,'w') as fp:  
        for confDict in modified_confs:
            ref_dir = app
            out_dir = app 
            cur_line = ''
            is_ref = False
            template = copy.deepcopy(template_init)
            if not title_generated:
                for confName,confValue in confDict.iteritems():
                    fp.write(confName + '\t')
                fp.write('report')
                fp.write('\n')
                title_generated = True
             
            for confName, value in confDict.iteritems():
                template[confName] = value
                out_dir += '_' + get_short_name(confName)+'-'+str(value).replace('/','-')
                cur_line +=str(value) + '\t'
                #replace the value of ref item to get ref dir
                if confName == ref_item:
                    ref_dir += '_' + get_short_name(confName)+'-'+str(ref_value).replace('/','-')
                    if ref_value == value:
                        is_ref = True
                else: 
                    ref_dir += '_' + get_short_name(confName)+'-'+str(value).replace('/','-')
            confDict['is_ref'] = is_ref
            if not is_ref:
                template['reference_dir'] = ref_dir 
                confDict['report_path'] = out_dir + '/test_report.txt'
                fp.write(cur_line)
                fp.write(out_dir + '/test_report.txt')
                fp.write('\n')
            template['out_dir'] = out_dir
            output_json_name = 'test-config-debug/' + out_dir + '.json'
             
            if getJson_only:
                result_list.append([output_json_name,is_ref])
            else:
                result_list.append([output_json_name, template])
    dict2json({ 'creator': json_path, 'application': app, 'cases_info' : modified_confs}, cases_info_json)
    
    return result_list

def get_short_name(fullname):
    
    short_name = ''
    if len(fullname.split('_')) > 1:
        for i in fullname.split('_'):
            short_name += i[0]
    else:
        short_name = fullname[0:3]
    return short_name

def conf_pre_process(itemDict):
           
    if itemDict.has_key('accuracy_level'):
        accuracy_level = itemDict['accuracy_level'] 
        if accuracy_level == 'fwd':
            itemDict['prototxt_type'] = 'train_val'
            itemDict['forward_only'] = True
        elif accuracy_level == 'bwd':
            itemDict['prototxt_type'] = 'train_val'
            itemDict['forward_only'] = False
        elif accuracy_level == 'train' :
            itemDict['prototxt_type'] = 'solver'            
        itemDict.pop('accuracy_level') 

    if itemDict.has_key('prototxt_type') and itemDict['prototxt_type'] == 'solver':
        itemDict.pop('forward_only',-1)
    
    if itemDict.has_key('batch_size') and itemDict['batch_size'] == 'default':
        if itemDict['topology'] == 'bvlc_alexnet':
            itemDict['batch_size'] = 256
        elif itemDict['topology'] == 'bvlc_googlenet':
            itemDict['batch_size'] = 32 
        elif itemDict['topology'] == 'googlenet_v2':
            itemDict['batch_size'] = 32 
        elif itemDict['topology'] == 'resnet_50':
            itemDict['batch_size'] = 50
        elif itemDict['topology'] == 'vgg_19':
            itemDict['batch_size'] = 64 
        elif itemDict['topology'] == 'ssd':
            itemDict['batch_size'] = 32
        elif itemDict['topology'] == 'googlenet_v3':
            itemDict['batch_size'] = 64 
        
    
    return itemDict
