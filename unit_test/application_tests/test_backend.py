import json, os,sys
SCRIPT_HOME = os.path.dirname(os.path.realpath(__file__))
print SCRIPT_HOME
sys.path.insert(1, os.path.join(os.path.dirname(__file__), '../../..'))
from backends import backends_factory
from utils.io import json2obj
config = {}
with open(os.path.join(os.path.dirname(__file__), '../../config-template.json'), 'r') as fp:
    config = json2obj(fp)

backend_class = backends_factory(config.backend)
backend = backend_class()
img = ['/home/xiaoyihe/chainer-SSD/img/fish-bike.jpg','/home/xiaoyihe/chainer-SSD/img/dog.jpg']
backend.prepare_infer(img, config)
backend.infer()

