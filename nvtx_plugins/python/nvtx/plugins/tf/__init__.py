# ! /usr/bin/python
# -*- coding: utf-8 -*-

# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ctypes

from nvtx.plugins.c_extensions import tensorflow_nvtx_lib
from nvtx.plugins.tf.ext_utils import load_library

nvtx_tf_ops = load_library(tensorflow_nvtx_lib)

from nvtx.plugins.tf import keras
from nvtx.plugins.tf import ops


__all__ = [
    "nvtx_tf_ops",

    # sub packages
    "keras",
    "ops"
]
