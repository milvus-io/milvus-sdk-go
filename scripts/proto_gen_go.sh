#!/usr/bin/env bash
SCRIPTS_DIR=$(dirname "$0")

PROTO_DIR=$SCRIPTS_DIR/../internal/proto
MILVUS_PROTO_DIR=$SCRIPTS_DIR/../internal/milvus-proto

PROGRAM=$(basename "$0")
GOPATH=$(go env GOPATH)

if [ -z $GOOGLE_PROTOPATH ]; then
    printf "Error: path to google proto not defined, please export GOOGLE_PROTOPATH before running this script"
    exit 1
fi

if [ -z $GOPATH ]; then
    printf "Error: GOPATH cannot be found, please set it before running this script"
    exit 1
fi

case ":$PATH:" in
    *":$GOPATH/bin:"*) ;;
    *) export PATH="$GOPATH/bin:$PATH";;
esac

echo "updating module-proto submodule"
git submodule update --init

echo "using protoc-gen-go: $(which protoc-gen-go)"

mkdir -p ${PROTO_DIR}/common
mkdir -p ${PROTO_DIR}/server
mkdir -p ${PROTO_DIR}/milvus

protoc --proto_path=${MILVUS_PROTO_DIR}/proto \
    --proto_path=${GOOGLE_PROTOPATH} \
    --go_opt="Mmilvus.proto=github.com/milvus-io/milvus-sdk-go/v2/internal/proto/server;server" \
    --go_opt=Mcommon.proto=github.com/milvus-io/milvus-sdk-go/v2/internal/proto/common \
    --go_opt=Mschema.proto=github.com/milvus-io/milvus-sdk-go/v2/internal/proto/schema \
    --go_out=plugins=grpc,paths=source_relative:${PROTO_DIR}/server ${MILVUS_PROTO_DIR}/proto/milvus.proto
protoc --proto_path=${MILVUS_PROTO_DIR}/proto \
    --proto_path=${GOOGLE_PROTOPATH} \
    --go_opt=Mmilvus.proto=github.com/milvus-io/milvus-sdk-go/v2/internal/proto/server \
    --go_opt="Mcommon.proto=github.com/milvus-io/milvus-sdk-go/v2/internal/proto/common;common" \
    --go_opt=Mschema.proto=github.com/milvus-io/milvus-sdk-go/v2/internal/proto/schema \
    --go_out=plugins=grpc,paths=source_relative:${PROTO_DIR}/common ${MILVUS_PROTO_DIR}/proto/common.proto
protoc --proto_path=${MILVUS_PROTO_DIR}/proto \
    --proto_path=${GOOGLE_PROTOPATH} \
    --go_opt=Mmilvus.proto=github.com/milvus-io/milvus-sdk-go/v2/internal/proto/server \
    --go_opt=Mcommon.proto=github.com/milvus-io/milvus-sdk-go/v2/internal/proto/common \
    --go_opt="Mschema.proto=github.com/milvus-io/milvus-sdk-go/v2/internal/proto/schema;schema" \
    --go_out=plugins=grpc,paths=source_relative:${PROTO_DIR}/schema ${MILVUS_PROTO_DIR}/proto/schema.proto

