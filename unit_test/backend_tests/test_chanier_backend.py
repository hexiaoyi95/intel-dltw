import json, os
from backends import backends_factory
config = {}
with open(os.path.join(os.path.dirname(__file__), '../../config-template.json'), 'r') as fp:
    config = json.load(fp)

backend_class = backends_factory(config['backend'])
backend = backend_class()
image_names = []
backend.prepare_detect(image_names,config)
backend.infer()
