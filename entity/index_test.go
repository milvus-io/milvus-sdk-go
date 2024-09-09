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

package entity

import (
	"fmt"
	"math/rand"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestGenericIndex(t *testing.T) {
	name := fmt.Sprintf("generic_index_%d", rand.Int())
	gi := NewGenericIndex(name, IvfFlat, map[string]string{
		tMetricType: string(IP),
	})
	assert.Equal(t, name, gi.Name())
	assert.EqualValues(t, IvfFlat, gi.Params()[tIndexType])
}

func TestAddRadius(t *testing.T) {
	params := newBaseSearchParams()
	params.AddRadius(10)
	assert.Equal(t, params.Params()["radius"], float64(10))
}

func TestAddRangeFilter(t *testing.T) {
	params := newBaseSearchParams()
	params.AddRangeFilter(20)
	assert.Equal(t, params.Params()["range_filter"], float64(20))
}

func TestPageRetainOrder(t *testing.T) {
	params := newBaseSearchParams()
	params.AddPageRetainOrder(true)
	assert.Equal(t, params.Params()["page_retain_order"], true)
}

func TestIndexGPUCagra(t *testing.T) {
	t.Run("index", func(t *testing.T) {
		index, err := NewIndexGPUCagra(L2, 64, 64)
		require.NoError(t, err)
		require.NotNil(t, index)

		assert.Equal(t, "GPUCagra", index.Name())
		assert.Equal(t, GPUCagra, index.IndexType())
		assert.False(t, index.SupportBinary())

		params := index.Params()

		metricType, ok := params["metric_type"]
		require.True(t, ok)
		assert.Equal(t, string(L2), metricType)

		indexType, ok := params["index_type"]
		require.True(t, ok)
		assert.Equal(t, string(GPUCagra), indexType)

		_, err = NewIndexGPUCagra(L2, 32, 64)
		assert.Error(t, err)
	})

	t.Run("search_param", func(t *testing.T) {
		sp, err := NewIndexGPUCagraSearchParam(
			64,
			1,
			0,
			0,
			4,
		)
		require.NoError(t, err)
		require.NotNil(t, sp)

		params := sp.Params()
		itopkSize, ok := params["itopk_size"]
		require.True(t, ok)
		assert.EqualValues(t, 64, itopkSize)
		searchWidth, ok := params["search_width"]
		require.True(t, ok)
		assert.EqualValues(t, 1, searchWidth)
		maxIterations, ok := params["max_iterations"]
		require.True(t, ok)
		assert.EqualValues(t, 0, maxIterations)
		minIterations, ok := params["min_iterations"]
		require.True(t, ok)
		assert.EqualValues(t, 0, minIterations)
		teamSize, ok := params["team_size"]
		require.True(t, ok)
		assert.EqualValues(t, 4, teamSize)

		_, err = NewIndexGPUCagraSearchParam(
			64,
			1,
			0,
			0,
			3,
		)
		assert.Error(t, err)
	})
}

func TestIndexGPUBruteForce(t *testing.T) {
	t.Run("index", func(t *testing.T) {
		index, err := NewIndexGPUBruteForce(L2)
		require.NoError(t, err)
		require.NotNil(t, index)

		assert.Equal(t, "GPUBruteForce", index.Name())
		assert.Equal(t, GPUBruteForce, index.IndexType())
		assert.False(t, index.SupportBinary())

		params := index.Params()

		metricType, ok := params["metric_type"]
		require.True(t, ok)
		assert.Equal(t, string(L2), metricType)

		indexType, ok := params["index_type"]
		require.True(t, ok)
		assert.Equal(t, string(GPUBruteForce), indexType)
	})

	t.Run("search_param", func(t *testing.T) {
		sp, err := NewIndexGPUBruteForceSearchParam()
		assert.NoError(t, err)
		assert.NotNil(t, sp)
	})
}
