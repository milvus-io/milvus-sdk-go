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

func TestMakeSearchQueryOption(t *testing.T) {
	c := &entity.Collection{
		ID:               999,
		ConsistencyLevel: entity.CL_STRONG,
	}

	t.Run("strong consistency", func(t *testing.T) {
		opt, err := MakeSearchQueryOption(c)
		assert.Nil(t, err)
		assert.NotNil(t, opt)
		expected := &SearchQueryOption{
			ConsistencyLevel:   entity.CL_STRONG,
			GuaranteeTimestamp: StrongTimestamp,
		}
		assert.Equal(t, expected, opt)
	})

	t.Run("session consistency", func(t *testing.T) {
		opt, err := MakeSearchQueryOption(c, WithSearchQueryConsistencyLevel(entity.CL_SESSION))
		assert.Nil(t, err)
		assert.NotNil(t, opt)
		expected := &SearchQueryOption{
			ConsistencyLevel:   entity.CL_SESSION,
			GuaranteeTimestamp: EventuallyTimestamp,
		}
		assert.Equal(t, expected, opt)

		tsm.set(c.ID, 99)
		opt, err = MakeSearchQueryOption(c, WithSearchQueryConsistencyLevel(entity.CL_SESSION))
		assert.Nil(t, err)
		assert.NotNil(t, opt)
		expected = &SearchQueryOption{
			ConsistencyLevel:   entity.CL_SESSION,
			GuaranteeTimestamp: 99,
		}
		assert.Equal(t, expected, opt)
	})

	t.Run("bounded consistency", func(t *testing.T) {
		opt, err := MakeSearchQueryOption(c, WithSearchQueryConsistencyLevel(entity.CL_BOUNDED))
		assert.Nil(t, err)
		assert.NotNil(t, opt)
		expected := &SearchQueryOption{
			ConsistencyLevel:   entity.CL_BOUNDED,
			GuaranteeTimestamp: BoundedTimestamp,
		}
		assert.Equal(t, expected, opt)
	})

	t.Run("eventually consistency", func(t *testing.T) {
		opt, err := MakeSearchQueryOption(c, WithSearchQueryConsistencyLevel(entity.CL_EVENTUALLY))
		assert.Nil(t, err)
		assert.NotNil(t, opt)
		expected := &SearchQueryOption{
			ConsistencyLevel:   entity.CL_EVENTUALLY,
			GuaranteeTimestamp: EventuallyTimestamp,
		}
		assert.Equal(t, expected, opt)
	})

	t.Run("customized consistency", func(t *testing.T) {
		opt, err := MakeSearchQueryOption(c, WithSearchQueryConsistencyLevel(entity.CL_CUSTOMIZED), WithGuaranteeTimestamp(100))
		assert.Nil(t, err)
		assert.NotNil(t, opt)
		expected := &SearchQueryOption{
			ConsistencyLevel:   entity.CL_CUSTOMIZED,
			GuaranteeTimestamp: 100,
		}
		assert.Equal(t, expected, opt)
	})

	t.Run("guarantee timestamp sanity check", func(t *testing.T) {
		_, err := MakeSearchQueryOption(c, WithSearchQueryConsistencyLevel(entity.CL_STRONG), WithGuaranteeTimestamp(100))
		assert.Error(t, err)
	})
}
