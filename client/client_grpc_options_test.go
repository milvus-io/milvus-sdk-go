// Copyright (C) 2019-2021 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License.

package client

import (
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
	"github.com/milvus-io/milvus-sdk-go/v2/internal/proto/common"
	"github.com/milvus-io/milvus-sdk-go/v2/internal/proto/server"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestCreateCollectionWithConsistencyLevel(t *testing.T) {
	opt := WithConsistencyLevel(entity.ConsistencyLevel(entity.CL_BOUNDED))
	assert.NotNil(t, opt)
	req := &server.CreateCollectionRequest{}

	assert.NotPanics(t, func() {
		opt(req)
	})

	assert.Equal(t, common.ConsistencyLevel_Bounded, req.GetConsistencyLevel())
}

func TestLoadCollectionWithReplicaNumber(t *testing.T) {
	opt := WithReplicaNumber(testMultiReplicaNumber)
	assert.NotNil(t, opt)
	req := &server.LoadCollectionRequest{}

	assert.NotPanics(t, func() {
		opt(req)
	})

	assert.Equal(t, testMultiReplicaNumber, req.GetReplicaNumber())
}
