import mxnet as mx
import ctypes
from mxnet.base import c_str, check_call, string_types
import os


dll_path = os.path.abspath('libdummy_op.so')
MXNET_LIB_CTYPES = ctypes.CDLL(dll_path, ctypes.RTLD_GLOBAL)

help(MXNET_LIB_CTYPES)
