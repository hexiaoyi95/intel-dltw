#!/usr/bin/env python
# encoding: utf-8
import os, sys
import logging
import glob
import numpy as np
import skimage.io
import json
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
