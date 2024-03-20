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

func TestSliceSparseEmbedding(t *testing.T) {
	t.Run("normal_case", func(t *testing.T) {

		length := 1 + rand.Intn(5)
		positions := make([]uint32, length)
		values := make([]float32, length)
		for i := 0; i < length; i++ {
			positions[i] = uint32(i)
			values[i] = rand.Float32()
		}
		se, err := NewSliceSparseEmbedding(positions, values)
		require.NoError(t, err)

		assert.EqualValues(t, length, se.Dim())
		assert.EqualValues(t, length, se.Len())

		bs := se.Serialize()
		nv, err := deserializeSliceSparceEmbedding(bs)
		require.NoError(t, err)

		for i := 0; i < length; i++ {
			pos, val, ok := se.Get(i)
			require.True(t, ok)
			assert.Equal(t, positions[i], pos)
			assert.Equal(t, values[i], val)

			npos, nval, ok := nv.Get(i)
			require.True(t, ok)
			assert.Equal(t, positions[i], npos)
			assert.Equal(t, values[i], nval)
		}

		_, _, ok := se.Get(-1)
		assert.False(t, ok)
		_, _, ok = se.Get(length)
		assert.False(t, ok)
	})

	t.Run("position values not match", func(t *testing.T) {
		_, err := NewSliceSparseEmbedding([]uint32{1}, []float32{})
		assert.Error(t, err)
	})

}

func TestColumnSparseEmbedding(t *testing.T) {
	columnName := fmt.Sprintf("column_sparse_embedding_%d", rand.Int())
	columnLen := 8 + rand.Intn(10)

	v := make([]SparseEmbedding, 0, columnLen)
	for i := 0; i < columnLen; i++ {
		length := 1 + rand.Intn(5)
		positions := make([]uint32, length)
		values := make([]float32, length)
		for j := 0; j < length; j++ {
			positions[j] = uint32(j)
			values[j] = rand.Float32()
		}
		se, err := NewSliceSparseEmbedding(positions, values)
		require.NoError(t, err)
		v = append(v, se)
	}
	column := NewColumnSparseVectors(columnName, v)

	t.Run("test column attribute", func(t *testing.T) {
		assert.Equal(t, columnName, column.Name())
		assert.Equal(t, FieldTypeSparseVector, column.Type())
		assert.Equal(t, columnLen, column.Len())
		assert.EqualValues(t, v, column.Data())
	})

	t.Run("test column field data", func(t *testing.T) {
		fd := column.FieldData()
		assert.NotNil(t, fd)
		assert.Equal(t, fd.GetFieldName(), columnName)
	})

	t.Run("test column value by idx", func(t *testing.T) {
		_, err := column.ValueByIdx(-1)
		assert.Error(t, err)
		_, err = column.ValueByIdx(columnLen)
		assert.Error(t, err)

		_, err = column.Get(-1)
		assert.Error(t, err)
		_, err = column.Get(columnLen)
		assert.Error(t, err)

		for i := 0; i < columnLen; i++ {
			v, err := column.ValueByIdx(i)
			assert.NoError(t, err)
			assert.Equal(t, column.vectors[i], v)
			getV, err := column.Get(i)
			assert.NoError(t, err)
			assert.Equal(t, v, getV)
		}
	})
}
