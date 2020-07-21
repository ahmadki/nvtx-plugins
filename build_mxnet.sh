rm -r libdummy_op.so
g++ -shared -fPIC -std=c++11 nvtx_plugins/cc/mxnet/dummy_op.cc -o libdummy_op.so -I $MXNET_HOME/include -I /usr/local/cuda/include/ -D MSHADOW_USE_CBLAS
