#!/usr/bin/env python
# encoding: utf-8
import os, sys
import logging
import glob
import numpy as np
import skimage.io
import json
def get_input_list(input_path, img_num = sys.maxint):
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
        while len(input_list) < img_num:
            input_list += input_list
        # only one batch here
        input_list = input_list[:img_num]
        logging.debug("Got {0:d} images".format(len(input_list)))
    else:
        if os.path.exists(input_path):
            logging.debug("Got one image: %s" % input_path)
            input_list.append(input_path)
        else:
            raise Exception("Invalid input path")
    return input_list


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
    with open(json_path, 'r') as fp:
        data = fp.read().replace('\n', '')
    obj = json.loads(data, object_hook=lambda d: namedtuple('X', d.keys())(*d.values()))
    return obj


