import numpy as np
import sys
#insert the intel-caffe's path
sys.path.append('dl-frameworks/dl_framework-intel_caffe/python')
import caffe


caffe.set_mode_cpu()
caffe.set_random_seed(0)

net = caffe.Net('dl-models/test.prototxt', caffe.TRAIN ,engine= 'CAFFE')
data_shape = net.blobs['data'].data.shape
top_diff = {}
#fake_input = np.random.random_sample(data_shape) * 2 - 1
#np.save('fake_input',fake_input)
input_data = np.load('out/convergency/bvlc_alexnet/ref/conv5/conv5_conv5_data.npy')
net.blobs['data'].data[:] = input_data
net.forward()
#for i in xrange(len(net.outputs)):
#    #diff = np.zeros_like(net.blobs[net.outputs[i]].data)
#    #diff += 100
#    diff = np.random.random_sample(net.blobs[net.outputs[i]].data.shape)*10 - 1
#    np.save('diff_{}'.format(i),diff)
#    #diff = np.random.random_sample(net.blobs[net.outputs[i]].diff.shape) * 2 - 1
#    #print net.outputs[i]
#    #diff = np.load('out/convergency/bvlc_alexnet/ref/pool5/pool5_diff.npy')	
#    top_diff[net.outputs[i]] = diff[...]
#
top_diff['pool5'] = np.load('out/convergency/bvlc_alexnet/ref/pool5_pool5_diff.npy')

net.backward(**top_diff)


#net.backward()
default = net.blobs['data'].diff
print default
#default = net.blobs['inception_5b/output'].diff

with open( 'default.txt','w') as f:
    f.write(str(default))
with open( 'ref_data_data.txt','w') as f:
    f.write(str(net.blobs['data'].data))
np.save('ref_pool\'s_bottom_diff',default)
np.save('ref_data_data',net.blobs['data'].data)
#inputDataIsEqual = np.allclose(net_MKL.blobs['data'].data, net.blobs['data'].data, rtol = 1e-02, atol = 1e-04)
#poolDataIsEqual = np.allclose(net_MKL.blobs['pool5'].data, net.blobs['pool5'].data, rtol = 1e-02, atol = 1e-04)
#fc6DataIsEqual = np.allclose(net_MKL.blobs['fc6'].data, net.blobs['fc6'].data, rtol = 1e-02, atol = 1e-04)
#outputDataIsEqual = np.allclose(net_MKL.blobs['prob'].data, net.blobs['prob'].data, rtol = 1e-02, atol = 1e-04)
#IsEqual = np.allclose(MKLDNN, default, rtol = 1e-02, atol = 1e-04)
#print 'input data is equal: ',inputDataIsEqual
#print 'output data is equal: ',outputDataIsEqual
#print 'pool5 data is equal: ',poolDataIsEqual
#print 'fc6 data is equal: ',fc6DataIsEqual
#print IsEqual
#print "input data sub"
#print net_MKL.blobs['data'].data - net.blobs['data'].data
#print 'output data sub'
#print net_MKL.blobs['prob'].data - net.blobs['prob'].data
#print "fc top data sub"
#print net_MKL.blobs['fc6'].data - net.blobs['fc6'].data
#print net_MKL.params['fc6'][0].data - net.params['fc6'][0].data
#print net_MKL.params['fc6'][1].data - net.params['fc6'][1].data
#print 'deference of bottom blob\'s diff in max pooling layer between MKLDNN and default:'
#print "diff subs"
#print MKLDNN - default
#print "fake input"
#print fake_input[0][0]
#
