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
INSTALL_PATH := $(PWD)/bin

all: check-protoc-version 

check-protoc:
ifeq (, $(shell which protoc))
	$(error "No protoc in PATH, consider doing apt-get install protoc")
else
	@echo "using $(shell which protoc)"
endif

install-tool:
	@mkdir -p $(INSTALL_PATH)
	@$(INSTALL_PATH)/golangci-lint --version 2>&1 1>/dev/null || (echo "Installing golangci-lint into ./bin/" && curl -sSfL https://raw.githubusercontent.com/golangci/golangci-lint/master/install.sh | sh -s -- -b $(INSTALL_PATH) v1.46.2)
	@$(INSTALL_PATH)/mockery --version 2>&1 1>/dev/null || (echo "Installing mockery v2.16.0 to ./bin/" && GOBIN=$(INSTALL_PATH)/ go install github.com/vektra/mockery/v2@v2.16.0)

check-protoc-version: check-protoc
	@(env bash $(PWD)/scripts/check_protoc_version.sh)

generate-proto: check-protoc-version
	@which protoc-gen-go 1>/dev/null || (echo "Installing protoc-gen-go" && go get github.com/golang/protobuf/protoc-gen-go@v1.5.2)
	@(env bash $(PWD)/scripts/proto_gen_go.sh)

generate-mockery: install-tool
	@echo "generating mockery milvus service server"
	@$(INSTALL_PATH)/mockery --srcpkg=github.com/milvus-io/milvus-proto/go-api/milvuspb --name=MilvusServiceServer --output=mocks --outpkg=mocks --with-expecter

static-check:
	@echo "Running $@ check:"
	@$(INSTALL_PATH)/golangci-lint cache clean
	@$(INSTALL_PATH)/golangci-lint run --timeout=30m --config ./.golangci.yml ./entity/...
	@$(INSTALL_PATH)/golangci-lint run --timeout=30m --config ./.golangci.yml ./client/...
	@$(INSTALL_PATH)/golangci-lint run --timeout=30m --config ./.golangci.yml ./internal/...

test-go:
	@echo "Running unit tests:"
	@(env bash $(PWD)/scripts/run_go_unittest.sh)

clean: 
	@echo "Cleaning up all generated file"
