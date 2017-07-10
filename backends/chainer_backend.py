import numpy as np
import utils
import logging
import os
import heapq
from skimage.transform import resize
os.environ['GLOG_minloglevel'] = '2'
import chainer
from chainer import serializers
from chainer import function_hooks
from chainer import functions as F
logger = logging.getLogger('root')
import sys
import bbox
import ssd
import pprint
from collections import OrderedDict
class ChainerBackend():
    def __init__(self, config):

        #print "Use Chainer as backend."
        #print chainer.__file__
        sys.path.insert(1,os.path.expanduser(os.path.dirname(config.model.topology)))
        net_py = os.path.basename(config.model.topology)
        net_module = __import__(os.path.splitext(net_py)[0])
        self.net = net_module.net()

        if hasattr(config.model, 'weight'):
            serializers.load_hdf5( os.path.expanduser(config.model.weight), self.net)
            logger.debug("Loaded weight")

        self.config = config
    def shuffle_inputs(self):

        utils.benchmark.shuffle_inputs(self.inputs)
        utils.benchmark.shuffle_inputs(self.input_names)

        self.set_net_input(self.inputs)

    def prepare_benchmark(self, config):
        """
        generate random input data
        """
        insize = self.net.insize
        self.inputs = np.random.uniform(-1,1,(config.batch_size,3,insize, insize)).astype('f')

    def prepare_infer(self,img_names, config):
        self.input_names = img_names
        self.inputs = self.image_preprocess(self.input_names, mean_value = config.mean_value)

    def prepare_classify(self, img_names, config):
        self.input_names = img_names
        self.inputs = self.image_preprocess(self.input_names, mean_value = config.mean_value)

    def image_preprocess(self,img_list,**kwargs):
        """
        Load images and transform input to chainer input
        """

        if "mean_file" in kwargs:
            mu = np.load(kwargs['mean_file']).mean(1).mean(1)
        elif "mean_value" in kwargs:
            mu = np.array(kwargs['mean_value'])


        inputs = [utils.io.load_image(im_f) for im_f in img_list]

        resize_dims = self.net.insize

        inputs = [ resize(img,(resize_dims,resize_dims))*255 - mu for img in inputs]

        inputs = [ img.transpose(2,0,1) for img in inputs ]

        return inputs

    def get_detection_output(self, nms_th=0.45, cls_th=0.6):

        output = dict()
        ps = [[] for i in range(len(self.input_names))]

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
                if scores[i] > cls_th:
                    cand.append(np.hstack([[label], [scores[i]], cand_loc[i]]))
                    ps[0].append((float(scores[i]), int(label), float(cand_loc[i][0]), float(cand_loc[i][1]), float(cand_loc[i][2]),float(cand_loc[i][3])))
        output[self.input_names[0]] = ps[0]

        return output

    def get_classify_output(self, topN = 5 ):
        """
        Return : dict for images prediction
        {
            'image_name', [ (label_1, prob_1), (label_2, prob_2) ]
        }
        """
        output = {}
        predictions = F.softmax(self.output)
        predictions = predictions.data
        for i, prediction in enumerate(predictions):
            top_inds = prediction.argsort()[::-1][:topN]  # reverse sort and take five largest items
            ps = []
            for top_ind in top_inds:
                ps.append((top_ind, float(prediction[top_ind])))
            output[self.input_names[i]] = ps

        return output

    def get_layer_accuracy_output(self):
        """
        Return:
            [
                {"variable_name": data, ... ,"variable_name": data},
                {"parameter_name": data, ... ,"parameter_name": data}
            ]
        """
        datas = {}
        weights = {}

        if self.output.creator is None:
            return
        cand_funcs = []
        seen_set = set()
        seen_vars = set()
        func_num = 0
        def add_cand(cand):
            if cand not in seen_set:
                heapq.heappush(cand_funcs,(-cand.rank, len(seen_set), cand))
                seen_set.add(cand)

        add_cand(self.output.creator)
        inputs = {}
        while cand_funcs:
            _,_, func = heapq.heappop(cand_funcs)
            func_num += 1
            outputs = [y() for y in func.outputs]
            num = len(datas)
            func_name = str(type(func)).split('.')[3]

            for i, y in enumerate(outputs):
                seen_vars.add(id(y))
                datas['%04d' % (func_num) + "_" + func_name + "_" +str(i + 1)  + "_data"] = y.data
                datas['%04d' % (func_num) + "_" + func_name + "_" +str(i + 1)  + "_diff"] = y.grad

            for i,x in enumerate(func.inputs):

                inputs['%04d' % (func_num) + "_" + func_name + "_" + str(i +1) ] = x

                if x.creator is not None:
                    add_cand(x.creator)

        for key in inputs:
            x = inputs[key]
            if x.name is not None and id(x) not in seen_vars:
                weights[key + "_params_" + x.name +"_diff"] = x.grad

        return datas, weights
    
    def get_layer_accuracy_output_debug(self, config):
        result = OrderedDict()
        if self.output.creator is None:
            return
        cand_funcs = []
        seen_set = set()
        seen_vars = set()
        func_num = -1
        def add_cand(cand):
            if cand not in seen_set:
                heapq.heappush(cand_funcs,(-cand.rank, len(seen_set), cand))
                seen_set.add(cand)

        add_cand(self.output.creator)
        inputs = {}
        while cand_funcs:
            _,_, func = heapq.heappop(cand_funcs)
            func_num += 1
            outputs = [y() for y in func.outputs]
            func_name = str(type(func)).split('.')[3]
            layer_result = list()
            for i, y in enumerate(outputs):
                seen_vars.add(id(y))
                #datas['%04d' % (func_num) + "_" + func_name + "_" +str(i + 1)  + "_data"] = y.data
                #datas['%04d' % (func_num) + "_" + func_name + "_" +str(i + 1)  + "_diff"] = y.grad
                if config.forward_only == False:
                    layer_result.append(["blob_{}".format(i),[y.data, y.grad]])
                else: 
                    layer_result.append(["blob_{}".format(i),[y.data]])
            result[ '%04d' % (func_num) + "_" + func_name] = layer_result
            inputs[ '%04d' % (func_num) + "_" + func_name] = list()
            for i,x in enumerate(func.inputs):

                inputs[ '%04d' % (func_num) + "_" + func_name].append(x) 

                if x.creator is not None:
                    add_cand(x.creator)

        for key in inputs:
            x_list = inputs[key]
            params = list()
            for x in x_list:
                if x.name is not None and id(x) not in seen_vars:
                    #weights[key + "_params_" + x.name +"_diff"] = x.grad
                    params.append(x)
            if len(params) > 0 and config.forward_only == False:
                #result[key].append(['params_data', [item.data for item in params]])
                result[key].append(['params_diff', [item.grad for item in params]])
                for item in params:
                    print item.grad
        reversed_result = OrderedDict()
        
        for i in sorted(result.keys(), key = lambda t:t.split('_')[0], reverse=True):
            reversed_result[i] = result[i]
        
        return reversed_result
            

    def get_layers_perf(self,direction):
        """
        Parameters:
            direction: forward or backward
        Return:
            list:
                [
                    [[function_name,time_ms]...[function_name,time_ms]],
                    total_time_ms
                ]
        """
        with chainer.function_hooks.TimerHook() as m:
            if direction == "forward":
                self.forward()
            elif direction == "backward":
                self.backward()
            call_his = m.call_history
            total_time_seconds = m.total_time()
        call = []
        for i,l in enumerate(call_his):
            func_type = type(l[0])
            #name = str(func_type).split(".")[3]
            name = i
            if direction == "backward":
                name = len(call_his) - i -1
            ms_time = l[1]* 1000
            call.append([name,ms_time])

        return [call,total_time_seconds * 1000]
    
    def layers(self,direction):

        with chainer.function_hooks.TimerHook() as m:
            self.prepare_benchmark(self.config) 
            if direction == "forward":
                self.forward()
            elif direction == "backward":
                self.backward()
            call_his = m.call_history
        layers = list()
        for i in call_his:
            layers.append(str(type(i[0])))
        #pprint.pprint( layers )
        return layers
    
    def get_layer_name(self, layer_id):
        return "To do"

    def infer(self):

        input_data = chainer.Variable(np.array(self.inputs,dtype=np.float32))
        self.output = self.net(input_data)

    def forward(self):

        input_data = chainer.Variable(np.array(self.inputs,dtype=np.float32))
        self.output = self.net(input_data)

    def backward(self):
        """
        first clear the grad ,then set the output's grad to 1 and do backward
        """
        self.net.cleargrads()
        self.output.grad = np.ones_like(self.output.data)
        self.output.backward(retain_grad = True)

    def get_layer_type(self,count):
        
        return "To do"




