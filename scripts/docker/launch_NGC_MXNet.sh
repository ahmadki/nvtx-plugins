#!/bin/bash

docker run -it --rm \
  --env PYTHONDONTWRITEBYTECODE=1 \
  --gpus=all \
  --ipc=host \
  -v "$(pwd):/workspace/nvtx" \
  --workdir /workspace/nvtx/ \
  nvcr.io/nvidia/mxnet:20.06-py3 /bin/bash
