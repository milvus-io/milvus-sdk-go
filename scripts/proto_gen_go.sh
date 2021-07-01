#!/usr/bin/env bash
SCRIPTS_DIR=$(dirname "$0")

PROTO_DIR=$SCRIPTS_DIR/../proto

PROGRAM=$(basename "$0")
GOPATH=$(go env GOPATH)

if [ -z $GOPATH ]; then
    printf "Error: GOPATH cannot be found, please set it before running this script"
    exit 1
fi

case ":$PATH:" in
    *":$GOPATH/bin:"*) ;;
    *) export PATH="$GOPATH/bin:$PATH";;
esac

echo "using protoc-gen-go: $(which protoc-gen-go)"

pushd ${PROTO_DIR}

mkdir -p common
mkdir -p schema
mkdir -p milvus


protoc --go_out=plugins=grpc,paths=source_relative:./milvus milvus.proto
protoc --go_out=plugins=grpc,paths=source_relative:./common common.proto
protoc --go_out=plugins=grpc,paths=source_relative:./schema schema.proto

popd

