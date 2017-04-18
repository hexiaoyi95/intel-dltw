import json, os,sys
SCRIPT_HOME = os.path.dirname(os.path.realpath(__file__))
print SCRIPT_HOME
sys.path.insert(1, os.path.join(os.path.dirname(__file__), '../..'))
from backends import backends_factory
from utils.io import json2obj
config = json2obj(os.path.join(os.path.dirname(__file__), 'test-chainer.json'))
image_names = ["/home/xiaoyihe/chainer-SSD/fish-bike.jpg"]
backend_class = backends_factory(config.backend)
backend = backend_class(config)
backend.prepare_infer(image_names,config)
backend.infer()
print backend.get_detection_output()


