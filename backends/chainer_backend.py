import numpy as np
import utils
import logging
import os
from skimage.transform import resize
os.environ['GLOG_minloglevel'] = '2'
import chainer
from chainer import serializers
logger = logging.getLogger('root')
import sys
sys.path.insert(1,"/home/xiaoyihe/chainer-SSD")
import ssd_net
import bbox
import ssd
class ChainerBackend():
    def __init__(self, config):
        print "Use Chainer as backend."
        self.net = ssd_net.SSD()
        serializers.load_npz(config.model.weight,self.net)

    def  shuffle_inputs(self):
        utils.benchmark.shuffle_inputs(self.inputs)
        utils.benchmark.shuffle_inputs(self.input_names)

        self.set_net_input(self.inputs)

    def prepare_infer(self,img_names, config):

        self.input_names = img_names
        self.inputs = self.image_preprocess(self.input_names, mean_value = config.mean_value)

    def image_preprocess(self,img_list,**kwargs):

        if "mean_file" in kwargs:
            mu = np.load(kwargs['mean_file']).mean(1).mean(1)
        elif "mean_value" in kwargs:
            mu = np.array(kwargs['mean_value'])


        inputs = [utils.io.load_image(im_f) for im_f in img_list]

        resize_dims = self.net.insize;

        inputs = [ resize(img,(resize_dims,resize_dims))*255 - mu for img in inputs]

        inputs = [ img.transpose(2,0,1)[::-1] for img in inputs ]

        return inputs

    def get_detection_output(self, nms_th=0.45, cls_th=0.6):
        prior = self.net.mbox_prior.astype(np.float32)
        loc = self.net.mbox_loc.data[0]
        conf = self.net.mbox_conf_softmax_reahpe.data[0]

        cand = []
        loc = ssd.decoder(loc, prior)
        for label in range(1, 21):
            #cand_score = np.where(conf[:, label] > cls_th)
            scores = conf[:, label]#[cand_score]
            cand_loc = loc#[cand_score]
            k = bbox.nms(cand_loc, scores, nms_th, 300)
            for i in k:

                cand.append(np.hstack([[label], [scores[i]], cand_loc[i]]))
        cand = np.array(cand)
        return cand



    def forward(self):

        input_data = chainer.Variable(np.array(self.inputs,dtype=np.float32))

        self.net(input_data,1,1,1,1)






