import json, os
from backends import backends_factory
config = {}
with open(os.path.join(os.path.dirname(__file__), 'test-config.json'), 'r') as fp:
    config = json.load(fp)

backend_class = backends_factory(config['backend'])
backend = backend_class()
backend.prepare_infer(config)
