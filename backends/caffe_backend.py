import caffe
import numpy as np
import utils
import logging
logger = logging.getLogger('root')
class CaffeBackend():
    def __init__(self):
        print "Use Caffe as backend."

    def  shuffle_inputs(self):
        utils.benchmark.shuffle_inputs(self.inputs)
        utils.benchmark.shuffle_inputs(self.input_names)

        self.set_net_input(self.inputs)

    def prepare_classify(self, config):
        if  (not hasattr(config, 'engine')) or (config.engine == "default"):
            self.net = caffe.Net(str(config.model.topology), str(config.model.weight), caffe.TEST)
        else:
            self.net = caffe.Net(str(config.model.topology), caffe.TEST, 0, None, str(config.model.weight), str(config.engine))

        self.reshape_by_batch_size(config.batch_size)

        self.input_names = utils.io.get_input_list(config.input_path, config.batch_size)
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
        output : list for images prediction
        [
            ['image_name', [ (label, prob),
                ....
                ]
            ]
        ]
        """
        datas = {out: self.net.blobs[out].data for out in self.net.outputs}
        data = datas['prob']
        output = []
        for i, prediction in enumerate(data):
            top_inds = prediction.argsort()[::-1][:topN]  # reverse sort and take five largest items
            ps = []
            for top_ind in top_inds:
                ps.append( (top_ind, prediction[top_ind]))
            output.append([self.input_names[i], ps])

        return output

    def infer(self):
        self.net.forward()
