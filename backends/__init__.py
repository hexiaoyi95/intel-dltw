import importlib
import sys

def backends_factory(backend_info):
    """
    backend_info :
        "python_path" : "/home/linxinan/workspace/caffe/intel/python",
        "class_path" : "backends.caffe_backend.CaffeBackend",
        "engine" : "default"
 
    return a backend class by target backend_info
    """

    sys.path.insert(1, backend_info.python_path)
    model = importlib.import_module(backend_info.class_path.rsplit('.', 1)[0])
    backend_class = getattr(model, backend_info.class_path.rsplit('.', 1)[1])

    return backend_class

