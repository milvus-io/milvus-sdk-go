#!/bin/bash

#~/workspace/milvus/core/cmake-build-debug/grpc_ep-prefix/src/grpc_ep/bins/opt/protobuf/protoc --proto_path=. --go_out=gen  milvus.proto

~/workspace/milvus/core/cmake-build-debug/grpc_ep-prefix/src/grpc_ep/bins/opt/protobuf/protoc -I . milvus.proto --go_out=plugins=grpc:gen

~/workspace/milvus/core/cmake-build-debug/grpc_ep-prefix/src/grpc_ep/bins/opt/protobuf/protoc -I . status.proto --go_out=plugins=grpc:gen

#~/workspace/milvus/core/cmake-build-debug/grpc_ep-prefix/src/grpc_ep/bins/opt/protobuf/protoc --proto_path=. --go_out=gen  status.proto

