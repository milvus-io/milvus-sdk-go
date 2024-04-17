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
	"encoding/binary"
	"fmt"
	"math"
	"sort"

	"github.com/cockroachdb/errors"
	schema "github.com/milvus-io/milvus-proto/go-api/v2/schemapb"
)

type SparseEmbedding interface {
	Dim() int // the dimension
	Len() int // the actual items in this vector
	Get(idx int) (pos uint32, value float32, ok bool)
	Serialize() []byte
	FieldType() FieldType
}

var _ SparseEmbedding = sliceSparseEmbedding{}
var _ Vector = sliceSparseEmbedding{}

type sliceSparseEmbedding struct {
	positions []uint32
	values    []float32
	dim       int
	len       int
}

func (e sliceSparseEmbedding) Dim() int {
	return e.dim
}

func (e sliceSparseEmbedding) Len() int {
	return e.len
}

func (e sliceSparseEmbedding) FieldType() FieldType {
	return FieldTypeSparseVector
}

func (e sliceSparseEmbedding) Get(idx int) (uint32, float32, bool) {
	if idx < 0 || idx >= int(e.len) {
		return 0, 0, false
	}
	return e.positions[idx], e.values[idx], true
}

func (e sliceSparseEmbedding) Serialize() []byte {
	row := make([]byte, 8*e.Len())
	for idx := 0; idx < e.Len(); idx++ {
		pos, value, _ := e.Get(idx)
		binary.LittleEndian.PutUint32(row[idx*8:], pos)
		binary.LittleEndian.PutUint32(row[idx*8+4:], math.Float32bits(value))
	}
	return row
}

// Less implements sort.Interce
func (e sliceSparseEmbedding) Less(i, j int) bool {
	return e.positions[i] < e.positions[j]
}

func (e sliceSparseEmbedding) Swap(i, j int) {
	e.positions[i], e.positions[j] = e.positions[j], e.positions[i]
	e.values[i], e.values[j] = e.values[j], e.values[i]
}

func deserializeSliceSparceEmbedding(bs []byte) (sliceSparseEmbedding, error) {
	length := len(bs)
	if length%8 != 0 {
		return sliceSparseEmbedding{}, errors.New("not valid sparse embedding bytes")
	}

	length = length / 8

	result := sliceSparseEmbedding{
		positions: make([]uint32, length),
		values:    make([]float32, length),
		len:       length,
	}

	for i := 0; i < length; i++ {
		result.positions[i] = binary.LittleEndian.Uint32(bs[i*8 : i*8+4])
		result.values[i] = math.Float32frombits(binary.LittleEndian.Uint32(bs[i*8+4 : i*8+8]))
	}
	return result, nil
}

func NewSliceSparseEmbedding(positions []uint32, values []float32) (SparseEmbedding, error) {
	if len(positions) != len(values) {
		return nil, errors.New("invalid sparse embedding input, positions shall have same number of values")
	}

	se := sliceSparseEmbedding{
		positions: positions,
		values:    values,
		len:       len(positions),
	}

	sort.Sort(se)

	if se.len > 0 {
		se.dim = int(se.positions[se.len-1]) + 1
	}

	return se, nil
}

var _ (Column) = (*ColumnSparseFloatVector)(nil)

type ColumnSparseFloatVector struct {
	ColumnBase

	vectors []SparseEmbedding
	name    string
}

// Name returns column name.
func (c *ColumnSparseFloatVector) Name() string {
	return c.name
}

// Type returns column FieldType.
func (c *ColumnSparseFloatVector) Type() FieldType {
	return FieldTypeSparseVector
}

// Len returns column values length.
func (c *ColumnSparseFloatVector) Len() int {
	return len(c.vectors)
}

// Get returns value at index as interface{}.
func (c *ColumnSparseFloatVector) Get(idx int) (interface{}, error) {
	if idx < 0 || idx >= c.Len() {
		return nil, errors.New("index out of range")
	}
	return c.vectors[idx], nil
}

// ValueByIdx returns value of the provided index
// error occurs when index out of range
func (c *ColumnSparseFloatVector) ValueByIdx(idx int) (SparseEmbedding, error) {
	var r SparseEmbedding // use default value
	if idx < 0 || idx >= c.Len() {
		return r, errors.New("index out of range")
	}
	return c.vectors[idx], nil
}

func (c *ColumnSparseFloatVector) FieldData() *schema.FieldData {
	fd := &schema.FieldData{
		Type:      schema.DataType_SparseFloatVector,
		FieldName: c.name,
	}

	dim := int(0)
	data := make([][]byte, 0, len(c.vectors))
	for _, vector := range c.vectors {
		row := make([]byte, 8*vector.Len())
		for idx := 0; idx < vector.Len(); idx++ {
			pos, value, _ := vector.Get(idx)
			binary.LittleEndian.PutUint32(row[idx*8:], pos)
			binary.LittleEndian.PutUint32(row[idx*8+4:], math.Float32bits(value))
		}
		data = append(data, row)
		if vector.Dim() > dim {
			dim = vector.Dim()
		}
	}

	fd.Field = &schema.FieldData_Vectors{
		Vectors: &schema.VectorField{
			Dim: int64(dim),
			Data: &schema.VectorField_SparseFloatVector{
				SparseFloatVector: &schema.SparseFloatArray{
					Dim:      int64(dim),
					Contents: data,
				},
			},
		},
	}
	return fd
}

func (c *ColumnSparseFloatVector) AppendValue(i interface{}) error {
	v, ok := i.(SparseEmbedding)
	if !ok {
		return fmt.Errorf("invalid type, expect SparseEmbedding interface, got %T", i)
	}
	c.vectors = append(c.vectors, v)

	return nil
}

func (c *ColumnSparseFloatVector) Data() []SparseEmbedding {
	return c.vectors
}

func NewColumnSparseVectors(name string, values []SparseEmbedding) *ColumnSparseFloatVector {
	return &ColumnSparseFloatVector{
		name:    name,
		vectors: values,
	}
}
