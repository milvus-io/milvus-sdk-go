#!/bin/bash

protoc -I . milvus.proto --go_out=plugins=grpc:gen

protoc -I . status.proto --go_out=plugins=grpc:gen
