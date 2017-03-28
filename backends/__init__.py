import importlib
import sys

def backends_factory(backend_info):
    """
    backend_info : ["backends.caffe_backend.CaffeBackend", "/opt/caffe/intel/python"]
    return a backend class by target backend_info
    """

    sys.path.insert(1, backend_info[1])
    model = importlib.import_module(backend_info[0].rsplit('.', 1)[0])
    backend_class = getattr(model, backend_info[0].rsplit('.', 1)[1])

    return backend_class

