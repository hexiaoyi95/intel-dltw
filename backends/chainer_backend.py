import numpy as np
import utils
import logging
import os
from skimage.transform import resize
os.environ['GLOG_minloglevel'] = '2'
import chainer
from chainer import serializers
from chainer import function_hooks
logger = logging.getLogger('root')
import sys
import bbox
import ssd
class ChainerBackend():
    def __init__(self, config):
        print "Use Chainer as backend."
        if config.model.net == "googlenet":
            sys.path.insert(1,os.path.expanduser(config.model.path))
            googlenet = __import__("googlenet")
            self.net = googlenet.GoogLeNet()
        elif config.model.net == "ssd":
            sys.path.insert(1,os.path.expanduser(config.model.path))
            ssd = __import__("ssd_net")
            self.net = ssd.SSD()
        else:
            raise Exception('Unsupported net type, choose googlenet or ssd')

        if hasattr(config.model, 'weight'):
            serializers.load_npz( os.path.expanduser(config.model.weight), self.net)
            logger.debug("Loaded weight")


    def  shuffle_inputs(self):

        utils.benchmark.shuffle_inputs(self.inputs)
        utils.benchmark.shuffle_inputs(self.input_names)

        self.set_net_input(self.inputs)

    def prepare_benchmark(self, config):
        insize = self.net.insize
        self.inputs = np.random.uniform(-1,1,(config.batch_size,3,insize, insize)).astype('f')

    def prepare_infer(self,img_names, config):

        self.input_names = img_names
        self.inputs = self.image_preprocess(self.input_names, mean_value = config.mean_value)

    def image_preprocess(self,img_list,**kwargs):

        if "mean_file" in kwargs:
            mu = np.load(kwargs['mean_file']).mean(1).mean(1)
        elif "mean_value" in kwargs:
            mu = np.array(kwargs['mean_value'])


        inputs = [utils.io.load_image(im_f) for im_f in img_list]

        resize_dims = self.net.insize

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

    def get_net_forward_perf(self):
        with chainer.function_hooks.TimerHook() as m:
            self.forward()
            call_his = m.call_history
            total_time_seconds = m.total_time()
            return [call_his,total_time_seconds]

    def get_net_backward_perf(self):
        with chainer.function_hooks.TimerHook() as m:
            self.backward()
            call_his = m.call_history
            total_time_seconds = m.total_time()
            return [call_his,total_time_seconds]

    def forward(self):

        input_data = chainer.Variable(np.array(self.inputs,dtype=np.float32))

        self.output = self.net(input_data)

    def backward(self):
        for i in xrange(len(self.output)):
            self.output[i].grad = np.ones_like(self.output[i].data)

	    self.output[0].backward()






