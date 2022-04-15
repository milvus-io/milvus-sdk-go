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
	"context"
	"testing"

	"github.com/golang/protobuf/proto"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
	"github.com/milvus-io/milvus-sdk-go/v2/internal/proto/common"
	"github.com/milvus-io/milvus-sdk-go/v2/internal/proto/server"
	"github.com/stretchr/testify/assert"
	tmock "github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"
)

func TestCollectionWithConsistencyLevel(t *testing.T) {
	opt := CollectionWithConsistencyLevel(entity.ConsistencyLevel(entity.CL_BOUNDED))
	assert.NotNil(t, opt)
	req := &server.CreateCollectionRequest{}

	assert.NotPanics(t, func() {
		opt.OptCreateCollection(req)
	})

	assert.Equal(t, common.ConsistencyLevel_Bounded, req.GetConsistencyLevel())
}

type mockCreateCollectionOpt struct {
	tmock.Mock
}

func (m *mockCreateCollectionOpt) OptCreateCollection(req *server.CreateCollectionRequest) {
	m.Called(req)
}

func TestCreateCollection_WithConsistencyLevel(t *testing.T) {
	ctx := context.Background()
	c := testClient(ctx, t)
	// default, all collection name returns false
	mock.delInjection(mHasCollection)

	ds := defaultSchema()
	shardsNum := int32(1)
	opt := &mockCreateCollectionOpt{}
	sch := ds.ProtoMessage()
	bs, err := proto.Marshal(sch)
	require.NoError(t, err)
	req := &server.CreateCollectionRequest{
		DbName:           "", // reserved fields, not used for now
		CollectionName:   ds.CollectionName,
		Schema:           bs,
		ShardsNum:        shardsNum,
		ConsistencyLevel: common.ConsistencyLevel_Strong,
	}
	opt.On("OptCreateCollection", req).Return()

	err = c.CreateCollection(ctx, ds, shardsNum, opt)
	assert.NoError(t, err)
	opt.AssertExpectations(t)
}
