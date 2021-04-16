#!/usr/bin/env bash
SCRIPTS_DIR=$(dirname "$0")

PROTO_DIR=$SCRIPTS_DIR/milvus/grpc/

PROGRAM=$(basename "$0")
GOPATH=$(go env GOPATH)

if [ -z $GOPATH ]; then
    printf "Error: the environment variable GOPATH is not set, please set it before running %s\n" $PROGRAM > /dev/stderr
    exit 1
fi

export PATH=${GOPATH}/bin:$PATH
echo `which protoc-gen-go`

# official go code ship with the crate, so we need to generate it manually.
pushd ${PROTO_DIR}

#printf ${PWD}
mkdir -p gen

${protoc} --go_out=plugins=grpc,paths=source_relative:./gen milvus.proto
${protoc} --go_out=plugins=grpc,paths=source_relative:./gen status.proto

popd
