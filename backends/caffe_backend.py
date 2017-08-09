import numpy as np
import utils
import logging
import os
from utils.benchmark import Timer
# os.environ['GLOG_minloglevel'] = '2'
import caffe
import inspect
from collections import OrderedDict
import pprint
import shutil
from caffe.proto import caffe_pb2
import google.protobuf.text_format as txtf

logger = logging.getLogger('root')
class CaffeBackend():
    def __init__(self, config):
        # "Use Caffe as self."
        # caffe constructor: network_file, phase, level, stages, weight, engine
        topology_path = os.path.expanduser(str(config.model.topology))


        if  (hasattr(config.backend, 'engine')) and (config.backend.engine != "default"):
            engine = str(config.backend.engine)
        else:
            engine = 'CAFFE'

        if hasattr(config.model, 'weight'):
            logger.debug("loading weights from: {}".format(config.model.weight))
            weight_path = os.path.expanduser(str(config.model.weight))
        else:
            weight_path = None

        if config.model.type == "test":
            phase = caffe.TEST
        else:
            phase = caffe.TRAIN

        caffe.set_mode_cpu()
        caffe.set_random_seed(0)

        if hasattr(config,'batch_size') and config.model.prototxt_type == 'train_val':
            topology_path = self.reshape_in_train_val( topology_path, config.batch_size, \
                config.out_dir,)

        if config.model.prototxt_type  == 'solver':
            logger.debug("using engine: {}".format(engine))
            modified_solver_path =  os.path.join( str(config.out_dir), 'modified_solver.prototxt')
            if not os.path.exists(os.path.dirname(modified_solver_path )):
                os.makedirs(os.path.dirname(modified_solver_path ))
            solver_params = caffe_pb2.SolverParameter()
            with open(config.model.topology) as f:
                s = f.read()
                txtf.Merge(s,solver_params)
            solver_params.engine = engine
            if hasattr(config, 'batch_size'):
                solver_params.net = self.reshape_in_train_val( str(solver_params.net), \
                    config.batch_size, config.out_dir) 
            with open( modified_solver_path, 'w') as fp:
                fp.write(str(solver_params))           
            self.solver = caffe.get_solver( modified_solver_path )
            self.net = self.solver.net
            if weight_path != None:
	    	    self.net.copy_from(weight_path)
        else:
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
    
    def reshape_in_train_val(self, topology_path, batch_size, out_dir):
        net_params = caffe_pb2.NetParameter()
        with open(topology_path) as f:
            s = f.read()
            txtf.Merge(s,net_params)
        input_layer = net_params.layer[0]
        if input_layer.type == 'ImageData':
            input_layer.image_data_param.batch_size = batch_size
        elif input_layer.type == 'Data' or input_layer.type == 'AnnotatedData':
            input_layer.data_param.batch_size = batch_size
        elif input_layer.type == 'HDF5Data':
            input_layer.hdf5_data_param.batch_size = batch_size
        elif input_layer.type == 'WindowData':
            input_layer.window_data_param.batch_size = batch_size
        elif input_layer.type == 'MemoryData':
            input_layer.memory_data_param.batch_size = batch_size
        else:
            logger.warn('failed to set the batch_size,using default one in prototxt') 
        modified_net = 'modified_train_val.prototxt'
        modified_topology_path = os.path.join(str(out_dir),modified_net)
        if not os.path.exists(os.path.dirname(modified_topology_path)):
            os.makedirs(os.path.dirname(modified_topology_path))
        with open( modified_topology_path, 'w') as fp:
           fp.write(str(net_params))
        
        return modified_topology_path

    def reshape_by_batch_size(self, batch_size):
        if self.get_layer_type(0) == 'Data':
            logger.info('caffe does not support reshape for layer type: Data')
            return

        logger.debug('reshaping to batch size: {}'.format(batch_size))
        #first reshape the top blobs of data layer
        for top_blob_name in self.net.top_names['data']:
            orig_shape = self.net.blobs[top_blob_name].data.shape
            target_shape = list(orig_shape[:])
            target_shape[0] = batch_size

            logger.debug('reshaping top blob [{}] of data layer from {} to {}'.format(top_blob_name, orig_shape, target_shape))
            self.net.blobs[top_blob_name].reshape(*target_shape)

        #then reshape the whole network
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
        # logger.debug('process layer: {} {}'.format(layer_id, direction))
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

        return timer.milliseconds()

    def get_layers_perf(self, direction):
        """
        return
            per layer forward or backward time: [(layer name, layer type, elapsed_time, FPS), ... ]
        """
        total_time = 0.0
        layers_perf = []
        layer_ids = range(len(self.get_layers()))
        if direction == 'backward':
            layer_ids = range(len(self.get_layers())-1, -1, -1)

        for layer_id in layer_ids:
            layer_time = self.get_layer_perf(layer_id, direction)
            total_time += layer_time
            layers_perf.append([layer_id, layer_time])

        return [layers_perf, total_time]

    def get_layer_accuracy_output(self):
        datas = OrderedDict()
        diffs = OrderedDict()
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

    def get_layer_accuracy_output_debug(self, config):
        result = OrderedDict()
        count = 0

        for layer_name,top_blob_names in self.net.top_names.iteritems():
            count +=1
            layer_result = list()
            for blob_name in top_blob_names:
                layer_id = self.get_layer_id(layer_name) 
                top_blob = self.net.blobs[blob_name]      
                    
                if hasattr(config,'forward_only') and config.forward_only:
                    data = top_blob.data.copy()
                    layer_result.append([blob_name + '_data', [data]])
                #diff = top_blob.diff.copy()
                #layer_result.append([blob_name,[data,diff]])
                elif config.model.prototxt_type == 'train_val':
                    diff = top_blob.diff.copy()
                    layer_result.append([blob_name + '_diff' , [diff]])
                    
                    
            try:
                paramater = self.net.params[layer_name]
            except:
                pass
            else:
                if config.model.prototxt_type == 'solver':
                    layer_result.append(['params_data',[item.data.copy() for item in paramater]])
                elif config.model.prototxt_type == 'train_val' and not config.forward_only:
                    layer_result.append(['params_diff',[item.diff.copy() for item in paramater]])
             

            result[layer_name] = layer_result
        #for fix the issue which MKL2017 optimization rule 3 make
        if config.model.type == 'train' and ( config.backend.engine == 'MKL2017' or \
                config.backend.engine == 'MKLDNN' ):
            if config.backend.engine == 'MKL2017':
                conv_type = 'MklConvolution'
            else:
                conv_type = 'Convolution'
            for layer_name,layer_result in result.iteritems():
                layer_id = self.get_layer_id(layer_name)
                for index,[blob_name, data_list] in enumerate(layer_result):
                    #blob_name=orig_name + '_x' + '_data' or '_diff',now we want to rm '_x'
                    if blob_name.split('_')[-2] == 'x':
                        blob_name = blob_name[:-7]+blob_name[-5:]
                        result[self.get_layer_name(layer_id)][index] = [blob_name,data_list]
                        if self.get_layer_type(layer_id-1) == conv_type \
                            and self.get_layer_type(layer_id) == 'BatchNorm':
                            result[self.get_layer_name(layer_id-1)][index] = [blob_name,data_list] 
        return result

    def clear_param_diffs(self):   
        self.net.clear_param_diffs()

    def get_layers(self):
        return self.net.layers

    def get_layer_id(self, layer_name):
        return list(self.net._layer_names).index(layer_name)

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

    def step(self, iter_num =1):
        self.solver.step(iter_num)

    def infer(self):
        self.net.forward()
    def backward(self):
        self.net.backward()
