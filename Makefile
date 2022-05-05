# Copyright (C) 2021 Zilliz. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under the License.

GO		?= go
PWD 	:= $(shell pwd)
GOPATH 	:= $(shell $(GO) env GOPATH)
PROTOC	:= $(shell which protoc)
PROTOC_VER := $(shell protoc --version)

all: check-protoc-version 

check-protoc:
ifeq (, $(shell which protoc))
	$(error "No protoc in PATH, consider doing apt-get install protoc")
else
	@echo "using $(shell which protoc)"
endif

check-protoc-version: check-protoc
	@(env bash $(PWD)/scripts/check_protoc_version.sh)

generate-proto: check-protoc-version
	@which protoc-gen-go 1>/dev/null || (echo "Installing protoc-gen-go" && go get github.com/golang/protobuf/protoc-gen-go@v1.5.2)
	@(env bash $(PWD)/scripts/proto_gen_go.sh)

static-check:
	@echo "Running $@ check:"
	@golangci-lint cache clean
	@golangci-lint run --timeout=30m --config ./.golangci.yml ./entity/...
	@golangci-lint run --timeout=30m --config ./.golangci.yml ./client/...
	@golangci-lint run --timeout=30m --config ./.golangci.yml ./internal/...
	@golangci-lint run --timeout=30m --config ./.golangci.yml ./tests/...

test-go:
	@echo "Running unit tests:"
	@(env bash $(PWD)/scripts/run_go_unittest.sh)

clean: 
	@echo "Cleaning up all generated file"
