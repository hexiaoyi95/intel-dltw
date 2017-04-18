import numpy as np
import utils
import logging
import os
os.environ['GLOG_minloglevel'] = '2'
import caffe
logger = logging.getLogger('root')
class CaffeBackend():
    def __init__(self, config):
        print "Use Caffe as backend."
        topology_path = os.path.expanduser(str(config.model.topology))
        weight_path = os.path.expanduser(str(config.model.weight))
        if  (not hasattr(config, 'engine')) or (config.engine == "default"):
            self.net = caffe.Net(topology_path, weight_path, caffe.TEST)
        else:
            self.net = caffe.Net(topology_path, caffe.TEST, 0, None, weight_path, str(config.engine))

    def  shuffle_inputs(self):
        utils.benchmark.shuffle_inputs(self.inputs)
        utils.benchmark.shuffle_inputs(self.input_names)

        self.set_net_input(self.inputs)

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
        for j in xrange(detections.shape[0]):

            ps = []
            det_index = detections[j, 0, :, 0]
            det_label = detections[j, 0, :, 1]
            det_conf = detections[j, 0, :, 2]
            det_xmin = detections[j, 0, :, 3]
            det_ymin = detections[j, 0, :, 4]
            det_xmax = detections[j, 0, :, 5]
            det_ymax = detections[j, 0, :, 6]

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
                img_index = top_index[k]
                xmin = top_xmin[k]
                ymin = top_ymin[k]
                xmax = top_xmax[k]
                ymax = top_ymax[k]
                score = top_conf[k]
                label = int(top_label_indices[k])
                #label_name = top_labels[k]
                ps.append((float(score), label, float(xmin), float(ymin) ,float(xmax), float(ymax)));
            output[self.input_names[j]] = ps

        return output

    def get_layer_accuracy_output(self):

        datas = {}
        weights = {}
        for key,value in self.net.blobs.iteritems():
            datas[key] = value.data

        for key,value in self.net.params.iteritems():
            for index in xrange(len(value)):
                key = key + '_' + str(index)
                param = value[index]
                weights[key] = param.data

        return datas,weights


    def infer(self):
        self.net.forward()

    def backward(self):
        top_diff = {}
        for i in xrange(len(self.net.outputs)):
            #print self.net.outputs[i]
            diff = np.zeros_like(self.net.blobs[self.net.outputs[i]].diff)
            diff += 100
            top_diff[self.net.outputs[i]] = diff[...]
        self.net.backward(**top_diff)

        #print self.net.blobs['conv1_1'].diff
