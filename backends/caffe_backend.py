import numpy as np
import utils
import logging
import os
from utils.benchmark import Timer
# os.environ['GLOG_minloglevel'] = '2'
import caffe
import inspect
logger = logging.getLogger('root')
class CaffeBackend():
    def __init__(self, config):
        # "Use Caffe as self."
        # caffe constructor: network_file, phase, level, stages, weight, engine
        topology_path = os.path.expanduser(str(config.model.topology))
        if  (hasattr(config, 'engine')) and (config.engine != "default"):
            engine = str(config.self.engine)
        else:
            engine = None

        if hasattr(config.model, 'weight'):
            weight_path = os.path.expanduser(str(config.model.weight))
        else:
            weight_path = None

        if config.application == "applications.performance":
            logger.debug("self for performance")
            phase = caffe.TRAIN
        else:
            phase = caffe.TEST

        caffe.set_mode_cpu()
        try:
            logger.debug("using engine: {}".format(engine))
            self.net = caffe.Net(topology_path, phase, weights=weight_path, engine=engine)
        except:
            self.net = caffe.Net(topology_path, phase, weights=weight_path)

    def shuffle_inputs(self):
        utils.benchmark.shuffle_inputs(self.inputs)
        utils.benchmark.shuffle_inputs(self.input_names)

        self.set_net_input(self.inputs)

    def prepare_benchmark(self, config):
        self.reshape_by_batch_size(config.batch_size)

    def prepare_classify(self,img_names, config):
        self.reshape_by_batch_size(config.batch_size)

        self.input_names = img_names
        self.inputs = self.image_preprocess(self.input_names, mean_value = config.mean_value)

        self.set_net_input(self.inputs)

    def image_preprocess(self, img_list, **kwargs):
        """
        Load images and transform input to caffe input
        Parameters
        ----------
        img_list  :   string
        kwargs      :   dict
        mean_file : string
        mean_value : [v1, v2, v3]

        Returns
        ----------
        imgs: list of ndarray
        """
        mu = None
        if "mean_file" in kwargs:
            mu = np.load(kwargs['mean_file']).mean(1).mean(1)
        elif "mean_value" in kwargs:
            mu = np.array(kwargs['mean_value'])
            # mu = np.array([104.00698793, 116.66876762, 122.67891434])
        channel_swap = [2,1,0]
        transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})

        transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
        transformer.set_transpose('data', (2,0,1)) # move image channels to outermost dimension
        transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
        transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
        transformer.set_channel_swap('data', channel_swap)  # swap channels from RGB to BGR

        #load
        inputs =[utils.io.load_image(im_f)
                 for im_f in img_list]

        #resize to crop_dims
        crop_dims = np.array(self.net.blobs['data'].data.shape[2:])
        inputs = [caffe.io.resize_image(img, crop_dims)
                  for img in inputs]

        #transform
        inputs = [transformer.preprocess(self.net.inputs[0], img)
              for img in inputs]

        return inputs


    def reshape_by_batch_size(self, batch_size):
        self.net.blobs['data'].reshape(batch_size,
                                           self.net.blobs['data'].data.shape[1],
                                           self.net.blobs['data'].data.shape[2],
                                           self.net.blobs['data'].data.shape[3]
                                           )
        self.net.reshape()

    def set_net_input(self, inputs):
        """
        inputs: list of preprocessed input
        """
        self.net.blobs['data'].data[...] = inputs

    def get_classify_output(self, topN=5):
        """
        output : dict for images prediction
        {
            'image_name', [ (label_1, prob_1), (label_2, prob_2) ]
        }
        """
        datas = {out: self.net.blobs[out].data for out in self.net.outputs}
        data = datas['prob']
        output = {}
        for i, prediction in enumerate(data):
            top_inds = prediction.argsort()[::-1][:topN]  # reverse sort and take five largest items
            ps = []
            for top_ind in top_inds:
                ps.append((top_ind, float(prediction[top_ind])))
            output[self.input_names[i]] = ps
        return output

    def get_detection_output(self, threshold=0.6):
        """
        output: dict for object detection
        {
          'image_name', [(conf_1, label_1, xmin_1, ymin_1 ,xmax_1, ymax_1),
             (conf_2, label_2, xmin_2, ymin_2, xmax_2, ymax_2) ... ]
        }
        """
        detections = self.net.blobs['detection_out'].data;
        output = {}
        ps =  [[] for i in range(len(self.input_names))]
        det_index = detections[0, 0, :, 0]
        det_label = detections[0, 0, :, 1]
        det_conf = detections[0, 0, :, 2]
        det_xmin = detections[0, 0, :, 3]
        det_ymin = detections[0, 0, :, 4]
        det_xmax = detections[0, 0, :, 5]
        det_ymax = detections[0, 0, :, 6]

        top_indices = [i for i, conf in enumerate(det_conf) if conf >= threshold]
        top_index = det_index[top_indices]
        top_conf = det_conf[top_indices]
        top_label_indices = det_label[top_indices].tolist()
        #top_labels = get_labelname(labelmap, top_label_indices)
        top_xmin = det_xmin[top_indices]
        top_ymin = det_ymin[top_indices]
        top_xmax = det_xmax[top_indices]
        top_ymax = det_ymax[top_indices]

        for k in xrange(top_conf.shape[0]):
            img_index = int(top_index[k])
            xmin = top_xmin[k]
            ymin = top_ymin[k]
            xmax = top_xmax[k]
            ymax = top_ymax[k]
            score = top_conf[k]
            label = int(top_label_indices[k])
            #label_name = top_labels[k]
            ps[img_index].append((float(score), label, float(xmin), float(ymin) ,float(xmax), float(ymax)));
        for i in xrange(len(self.input_names)):
            output[self.input_names[i]] = ps[i]

        return output

    def get_layer_perf(self,layer_id, direction):

        if direction == "forward":
            go_through = self.forward_layer
        elif direction == "backward":
            go_through = self.backward_layer
        else:
            raise Exception('Expect forward or backward but get {}'.format(direction))

        timer = Timer()

        timer.start()
        go_through(layer_id)
        timer.stop()

        return [self.get_layer_name(layer_id), timer.milliseconds()]

    def get_layers_perf(self, direction):
        """
        return
            per layer forward or backward time: [(layer name, layer type, elapsed_time, FPS), ... ]
        """
        total_time = 0.0
        layers_perf = []
        layer_ids = range(len(self.layers()))
        if direction == 'backward':
            layer_ids = range(len(self.layers())-1, -1, -1)

        for layer_id in layer_ids:
            layer_perf = self.get_layer_perf(layer_id, direction)
            total_time += layer_perf[1]
            layers_perf.append(layer_perf)

        return [layers_perf, total_time]

    def get_layer_accuracy_output(self):

        datas = {}
        diffs = {}
        count = 0
        for key,value in self.net.blobs.iteritems():
            count +=1
            datas['%04d' % (count) +"_" + key + "_data"] = value.data
            diffs['%04d' % (count) +"_" + key + "_diff"] = value.diff
        count = 0

        for key,value in self.net.params.iteritems():
            count +=1
            for index in xrange(len(value)):
                param_key = '%04d' % (count) +"_" + key  + "_params_" + str(index) + "_diff"
                param = value[index]
                diffs[param_key] = param.diff

        return datas,diffs

    def layers(self):
        return self.net.layers

    def get_layer_name(self, layer_id):
        return list(self.net._layer_names)[layer_id]

    def get_layer_type(self, layer_id):
        return list(self.net.layers)[layer_id].type

    def forward(self):
        self.net.forward()

    def forward_layer(self, layer_id):
        self.net._forward(layer_id, layer_id)

    def backward_layer(self, layer_id):
        self.net._backward(layer_id, layer_id)


    def infer(self):
        self.net.forward()

    def backward(self):
        top_diff = {}
        for i in xrange(len(self.net.outputs)):
            #print self.net.outputs[i]
            diff = np.zeros_like(self.net.blobs[self.net.outputs[i]].diff)
            diff += 1000
            top_diff[self.net.outputs[i]] = diff[...]
        self.net.backward(**top_diff)

        #print self.net.blobs['conv1_1'].diff
