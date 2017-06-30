import numpy as np
import sys
#insert the intel-caffe's path
sys.path.append('dl-frameworks/dl_framework-intel_caffe/python')
import caffe


caffe.set_mode_cpu()
caffe.set_random_seed(0)

#fake_input = np.load('fake_input.npy')
net_MKL = caffe.Net('dl-models/test.prototxt', caffe.TRAIN, engine = 'MKLDNN')

input_data = np.load('out/convergency/bvlc_alexnet/out/conv5/conv5_conv5_data.npy')
net_MKL.blobs['data'].data[:] = input_data
net_MKL.forward()
top_diff = {}
#for i in xrange(len(net_MKL.outputs)):
#    #diff = np.zeros_like(net_MKL.blobs[net_MKL.outputs[i]].data)
#    diff = np.load('diff_{}.npy'.format(i))
#    #diff = np.random.random_sample(net.blobs[net.outputs[i]].diff.shape) * 2 - 1
#    #print net.outputs[i]
#    #diff = np.load('out/convergency/bvlc_alexnet/ref/pool5/pool5_diff.npy')	
#    top_diff[net_MKL.outputs[i]] = diff
#
top_diff['pool5'] = np.load('out/convergency/bvlc_alexnet/out/cov5/conv5_conv5_data.npy')

net_MKL.backward(**top_diff)
net_MKL.backward()
MKLDNN = net_MKL.blobs['data'].diff
#MKLDNN = net_MKL.blobs['inception_5b/output'].diff
with open('MKLDNN.txt','w') as f:
    f.write(str(MKLDNN))
default = np.load('ref_pool\'s_bottom_diff.npy')
ref_data_data = np.load('ref_data_data.npy')
#print MKLDNN
#print default

inputDataIsEqual = np.allclose(net_MKL.blobs['data'].data, fake_input, rtol = 1e-02, atol = 1e-04)
#poolDataIsEqual = np.allclose(net_MKL.blobs['pool5'].data, net.blobs['pool5'].data, rtol = 1e-02, atol = 1e-04)
#fc6DataIsEqual = np.allclose(net_MKL.blobs['fc6'].data, net.blobs['fc6'].data, rtol = 1e-02, atol = 1e-04)
#outputDataIsEqual = np.allclose(net_MKL.blobs['prob'].data, net.blobs['prob'].data, rtol = 1e-02, atol = 1e-04)
IsEqual = np.allclose(MKLDNN, default, rtol = 1e-02, atol = 1e-04)
print 'input data is equal: ',inputDataIsEqual
#print 'output data is equal: ',outputDataIsEqual
#print 'pool5 data is equal: ',poolDataIsEqual
#print 'fc6 data is equal: ',fc6DataIsEqual
print IsEqual
#print "input data sub"
#print net_MKL.blobs['data'].data - fake_input
#print 'output data sub'
#print net_MKL.blobs['prob'].data - net.blobs['prob'].data
#print "fc top data sub"
#print net_MKL.blobs['fc6'].data - net.blobs['fc6'].data
#print net_MKL.params['fc6'][0].data - net.params['fc6'][0].data
#print net_MKL.params['fc6'][1].data - net.params['fc6'][1].data
#print 'deference of bottom blob\'s diff in max pooling layer between MKLDNN and default:'
#print "diff subs"
print MKLDNN - default
#print "fake input"
#print fake_input[0][0]
#
