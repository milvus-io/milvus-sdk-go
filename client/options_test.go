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
	"math/rand"
	"testing"

	"github.com/milvus-io/milvus-proto/go-api/v2/milvuspb"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
	"github.com/stretchr/testify/assert"
)

func TestCreateCollectionWithConsistencyLevel(t *testing.T) {
	opt := WithConsistencyLevel(entity.ConsistencyLevel(entity.ClBounded))
	assert.NotNil(t, opt)
	o := &createCollOpt{}

	assert.NotPanics(t, func() {
		opt(o)
	})

	assert.Equal(t, entity.ClBounded, o.ConsistencyLevel)
}

func TestCreateCollectionWithPartitionNum(t *testing.T) {
	partitionNum := rand.Int63n(1000) + 1
	opt := WithPartitionNum(partitionNum)
	assert.NotNil(t, opt)
	o := &createCollOpt{}

	assert.NotPanics(t, func() {
		opt(o)
	})

	assert.Equal(t, partitionNum, o.NumPartitions)
}

func TestLoadCollectionWithReplicaNumber(t *testing.T) {
	opt := WithReplicaNumber(testMultiReplicaNumber)
	assert.NotNil(t, opt)
	req := &milvuspb.LoadCollectionRequest{}

	assert.NotPanics(t, func() {
		opt(req)
	})

	assert.Equal(t, testMultiReplicaNumber, req.GetReplicaNumber())
}

func TestMakeSearchQueryOption(t *testing.T) {
	cli := &GrpcClient{
		cache: newMetaCache(),
	}
	c := &entity.Collection{
		Name:             "999",
		ConsistencyLevel: entity.ClStrong,
	}

	cInfo := collInfo{
		Name:             c.Name,
		ConsistencyLevel: c.ConsistencyLevel,
	}
	cli.cache.setCollectionInfo(c.Name, &cInfo)

	t.Run("strong consistency", func(t *testing.T) {
		opt, err := cli.makeSearchQueryOption(c.Name)
		assert.Nil(t, err)
		assert.NotNil(t, opt)
		expected := &SearchQueryOption{
			ConsistencyLevel:           entity.ClStrong,
			GuaranteeTimestamp:         StrongTimestamp,
			IgnoreGrowing:              false,
			ForTuning:                  false,
			UseDefaultConsistencyLevel: true,
		}
		assert.Equal(t, expected, opt)
	})

	t.Run("ignore growing", func(t *testing.T) {
		opt, err := cli.makeSearchQueryOption(c.Name, WithIgnoreGrowing())
		assert.Nil(t, err)
		assert.NotNil(t, opt)
		expected := &SearchQueryOption{
			ConsistencyLevel:           entity.ClStrong,
			GuaranteeTimestamp:         StrongTimestamp,
			IgnoreGrowing:              true,
			ForTuning:                  false,
			UseDefaultConsistencyLevel: true,
		}
		assert.Equal(t, expected, opt)
	})

	t.Run("for tuning", func(t *testing.T) {
		opt, err := cli.makeSearchQueryOption(c.Name, WithForTuning())
		assert.Nil(t, err)
		assert.NotNil(t, opt)
		expected := &SearchQueryOption{
			ConsistencyLevel:           entity.ClStrong,
			GuaranteeTimestamp:         StrongTimestamp,
			IgnoreGrowing:              false,
			ForTuning:                  true,
			UseDefaultConsistencyLevel: true,
		}
		assert.Equal(t, expected, opt)
	})

	t.Run("session consistency", func(t *testing.T) {
		opt, err := cli.makeSearchQueryOption(c.Name, WithSearchQueryConsistencyLevel(entity.ClSession))
		assert.Nil(t, err)
		assert.NotNil(t, opt)
		expected := &SearchQueryOption{
			ConsistencyLevel:           entity.ClSession,
			GuaranteeTimestamp:         EventuallyTimestamp,
			IgnoreGrowing:              false,
			ForTuning:                  false,
			UseDefaultConsistencyLevel: false,
		}
		assert.Equal(t, expected, opt)

		cli.cache.setSessionTs(c.Name, 99)
		opt, err = cli.makeSearchQueryOption(c.Name, WithSearchQueryConsistencyLevel(entity.ClSession))
		assert.Nil(t, err)
		assert.NotNil(t, opt)
		expected = &SearchQueryOption{
			ConsistencyLevel:           entity.ClSession,
			GuaranteeTimestamp:         99,
			IgnoreGrowing:              false,
			ForTuning:                  false,
			UseDefaultConsistencyLevel: false,
		}
		assert.Equal(t, expected, opt)
	})

	t.Run("bounded consistency", func(t *testing.T) {
		opt, err := cli.makeSearchQueryOption(c.Name, WithSearchQueryConsistencyLevel(entity.ClBounded))
		assert.Nil(t, err)
		assert.NotNil(t, opt)
		expected := &SearchQueryOption{
			ConsistencyLevel:           entity.ClBounded,
			GuaranteeTimestamp:         BoundedTimestamp,
			IgnoreGrowing:              false,
			ForTuning:                  false,
			UseDefaultConsistencyLevel: false,
		}
		assert.Equal(t, expected, opt)
	})

	t.Run("eventually consistency", func(t *testing.T) {
		opt, err := cli.makeSearchQueryOption(c.Name, WithSearchQueryConsistencyLevel(entity.ClEventually))
		assert.Nil(t, err)
		assert.NotNil(t, opt)
		expected := &SearchQueryOption{
			ConsistencyLevel:           entity.ClEventually,
			GuaranteeTimestamp:         EventuallyTimestamp,
			IgnoreGrowing:              false,
			ForTuning:                  false,
			UseDefaultConsistencyLevel: false,
		}
		assert.Equal(t, expected, opt)
	})

	t.Run("customized consistency", func(t *testing.T) {
		opt, err := cli.makeSearchQueryOption(c.Name, WithSearchQueryConsistencyLevel(entity.ClCustomized), WithGuaranteeTimestamp(100))
		assert.Nil(t, err)
		assert.NotNil(t, opt)
		expected := &SearchQueryOption{
			ConsistencyLevel:           entity.ClCustomized,
			GuaranteeTimestamp:         100,
			IgnoreGrowing:              false,
			ForTuning:                  false,
			UseDefaultConsistencyLevel: false,
		}
		assert.Equal(t, expected, opt)
	})

	t.Run("guarantee timestamp sanity check", func(t *testing.T) {
		_, err := cli.makeSearchQueryOption(c.Name, WithSearchQueryConsistencyLevel(entity.ClStrong), WithGuaranteeTimestamp(100))
		assert.Error(t, err)
	})
}

func TestWithUpdateResourceGroupConfig(t *testing.T) {
	req := &milvuspb.UpdateResourceGroupsRequest{}

	WithUpdateResourceGroupConfig("rg1", &entity.ResourceGroupConfig{
		Requests: &entity.ResourceGroupLimit{NodeNum: 1},
	})(req)
	WithUpdateResourceGroupConfig("rg2", &entity.ResourceGroupConfig{
		Requests: &entity.ResourceGroupLimit{NodeNum: 2},
	})(req)

	assert.Equal(t, 2, len(req.ResourceGroups))
	assert.Equal(t, int32(1), req.ResourceGroups["rg1"].Requests.NodeNum)
	assert.Equal(t, int32(2), req.ResourceGroups["rg2"].Requests.NodeNum)
}

func TestWithCreateResourceGroup(t *testing.T) {
	req := &milvuspb.CreateResourceGroupRequest{}

	WithCreateResourceGroupConfig(&entity.ResourceGroupConfig{
		Requests: &entity.ResourceGroupLimit{NodeNum: 1},
	})(req)
	assert.Equal(t, int32(1), req.Config.Requests.NodeNum)
}
