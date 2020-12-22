#!/bin/bash

/home/$USER/workspace/milvus/core/cmake-build-debug/grpc_ep-prefix/src/grpc_ep/bins/opt/protobuf/protoc -I . milvus.proto --go_out=plugins=grpc:gen

/home/$USER/workspace/milvus/core/cmake-build-debug/grpc_ep-prefix/src/grpc_ep/bins/opt/protobuf/protoc -I . status.proto --go_out=plugins=grpc:gen


