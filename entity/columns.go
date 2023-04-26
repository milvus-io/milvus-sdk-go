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
	"errors"
	"fmt"
	"math"

	schema "github.com/milvus-io/milvus-proto/go-api/schemapb"
)

//go:generate go run gen/gen.go

// Column interface field type for column-based data frame
type Column interface {
	Name() string
	Type() FieldType
	Len() int
	FieldData() *schema.FieldData
	AppendValue(interface{}) error
}

// Vector interface vector used int search
type Vector interface {
	Dim() int
	Serialize() []byte
	FieldType() FieldType
}

// FloatVector float32 vector wrapper.
type FloatVector []float32

// Dim returns vector dimension.
func (fv FloatVector) Dim() int {
	return len(fv)
}

// FieldType returns coresponding field type.
func (fv FloatVector) FieldType() FieldType {
	return FieldTypeFloatVector
}

// Serialize serializes vector into byte slice, used in search placeholder
// LittleEndian is used for convention
func (fv FloatVector) Serialize() []byte {
	data := make([]byte, 0, 4*len(fv)) // float32 occupies 4 bytes
	buf := make([]byte, 4)
	for _, f := range fv {
		binary.LittleEndian.PutUint32(buf, math.Float32bits(f))
		data = append(data, buf...)
	}
	return data
}

// BinaryVector []byte vector wrapper
type BinaryVector []byte

// Dim return vector dimension, note that binary vector is bits count
func (bv BinaryVector) Dim() int {
	return 8 * len(bv)
}

// Serialize just return bytes
func (bv BinaryVector) Serialize() []byte {
	return bv
}

// FieldType returns coresponding field type.
func (bv BinaryVector) FieldType() FieldType {
	return FieldTypeBinaryVector
}

var errFieldDataTypeNotMatch = errors.New("FieldData type not matched")

// IDColumns converts schema.IDs to corresponding column
// currently Int64 / string may be in IDs
func IDColumns(idField *schema.IDs, begin, end int) (Column, error) {
	var idColumn Column
	if idField == nil {
		return nil, errors.New("nil Ids from response")
	}
	switch field := idField.GetIdField().(type) {
	case *schema.IDs_IntId:
		if end >= 0 {
			idColumn = NewColumnInt64("", field.IntId.GetData()[begin:end])
		} else {
			idColumn = NewColumnInt64("", field.IntId.GetData()[begin:])
		}
	case *schema.IDs_StrId:
		if end >= 0 {
			idColumn = NewColumnVarChar("", field.StrId.GetData()[begin:end])
		} else {
			idColumn = NewColumnVarChar("", field.StrId.GetData()[begin:])
		}
	default:
		return nil, fmt.Errorf("unsupported id type %v", field)
	}
	return idColumn, nil
}

// FieldDataColumn converts schema.FieldData to Column, used int search result conversion logic
// begin, end specifies the start and end positions
func FieldDataColumn(fd *schema.FieldData, begin, end int) (Column, error) {
	switch fd.GetType() {
	case schema.DataType_Bool:
		data, ok := fd.GetScalars().GetData().(*schema.ScalarField_BoolData)
		if !ok {
			return nil, errFieldDataTypeNotMatch
		}
		if end < 0 {
			return NewColumnBool(fd.GetFieldName(), data.BoolData.GetData()[begin:]), nil
		}
		return NewColumnBool(fd.GetFieldName(), data.BoolData.GetData()[begin:end]), nil

	case schema.DataType_Int8:
		data, ok := fd.GetScalars().GetData().(*schema.ScalarField_IntData)
		if !ok {
			return nil, errFieldDataTypeNotMatch
		}
		values := make([]int8, 0, len(data.IntData.GetData()))
		for _, v := range data.IntData.GetData() {
			values = append(values, int8(v))
		}

		if end < 0 {
			return NewColumnInt8(fd.GetFieldName(), values[begin:]), nil
		}

		return NewColumnInt8(fd.GetFieldName(), values[begin:end]), nil

	case schema.DataType_Int16:
		data, ok := fd.GetScalars().GetData().(*schema.ScalarField_IntData)
		if !ok {
			return nil, errFieldDataTypeNotMatch
		}
		values := make([]int16, 0, len(data.IntData.GetData()))
		for _, v := range data.IntData.GetData() {
			values = append(values, int16(v))
		}
		if end < 0 {
			return NewColumnInt16(fd.GetFieldName(), values[begin:]), nil
		}

		return NewColumnInt16(fd.GetFieldName(), values[begin:end]), nil

	case schema.DataType_Int32:
		data, ok := fd.GetScalars().GetData().(*schema.ScalarField_IntData)
		if !ok {
			return nil, errFieldDataTypeNotMatch
		}
		if end < 0 {
			return NewColumnInt32(fd.GetFieldName(), data.IntData.GetData()[begin:]), nil
		}
		return NewColumnInt32(fd.GetFieldName(), data.IntData.GetData()[begin:end]), nil

	case schema.DataType_Int64:
		data, ok := fd.GetScalars().GetData().(*schema.ScalarField_LongData)
		if !ok {
			return nil, errFieldDataTypeNotMatch
		}
		if end < 0 {
			return NewColumnInt64(fd.GetFieldName(), data.LongData.GetData()[begin:]), nil
		}
		return NewColumnInt64(fd.GetFieldName(), data.LongData.GetData()[begin:end]), nil

	case schema.DataType_Float:
		data, ok := fd.GetScalars().GetData().(*schema.ScalarField_FloatData)
		if !ok {
			return nil, errFieldDataTypeNotMatch
		}
		if end < 0 {
			return NewColumnFloat(fd.GetFieldName(), data.FloatData.GetData()[begin:]), nil
		}
		return NewColumnFloat(fd.GetFieldName(), data.FloatData.GetData()[begin:end]), nil

	case schema.DataType_Double:
		data, ok := fd.GetScalars().GetData().(*schema.ScalarField_DoubleData)
		if !ok {
			return nil, errFieldDataTypeNotMatch
		}
		if end < 0 {
			return NewColumnDouble(fd.GetFieldName(), data.DoubleData.GetData()[begin:]), nil
		}
		return NewColumnDouble(fd.GetFieldName(), data.DoubleData.GetData()[begin:end]), nil

	case schema.DataType_String:
		data, ok := fd.GetScalars().GetData().(*schema.ScalarField_StringData)
		if !ok {
			return nil, errFieldDataTypeNotMatch
		}
		if end < 0 {
			return NewColumnString(fd.GetFieldName(), data.StringData.GetData()[begin:]), nil
		}
		return NewColumnString(fd.GetFieldName(), data.StringData.GetData()[begin:end]), nil

	case schema.DataType_VarChar:
		data, ok := fd.GetScalars().GetData().(*schema.ScalarField_StringData)
		if !ok {
			return nil, errFieldDataTypeNotMatch
		}
		if end < 0 {
			return NewColumnVarChar(fd.GetFieldName(), data.StringData.GetData()[begin:]), nil
		}
		return NewColumnVarChar(fd.GetFieldName(), data.StringData.GetData()[begin:end]), nil
	default:
		return nil, fmt.Errorf("unsupported data type %s", fd.GetType())
	}
}

// FieldDataColumn converts schema.FieldData to vector Column
func FieldDataVector(fd *schema.FieldData) (Column, error) {
	switch fd.GetType() {
	case schema.DataType_FloatVector:
		vectors := fd.GetVectors()
		x, ok := vectors.GetData().(*schema.VectorField_FloatVector)
		if !ok {
			return nil, errFieldDataTypeNotMatch
		}
		data := x.FloatVector.GetData()
		dim := int(vectors.GetDim())
		vector := make([][]float32, 0, len(data)/dim) // shall not have remanunt
		for i := 0; i < len(data)/dim; i++ {
			v := make([]float32, dim)
			copy(v, data[i*dim:(i+1)*dim])
			vector = append(vector, v)
		}
		return NewColumnFloatVector(fd.GetFieldName(), dim, vector), nil
	case schema.DataType_BinaryVector:
		vectors := fd.GetVectors()
		x, ok := vectors.GetData().(*schema.VectorField_BinaryVector)
		if !ok {
			return nil, errFieldDataTypeNotMatch
		}
		data := x.BinaryVector
		dim := int(vectors.GetDim())
		blen := dim / 8
		vector := make([][]byte, 0, len(data)/blen)
		for i := 0; i < len(data)/blen; i++ {
			v := make([]byte, blen)
			copy(v, data[i*blen:(i+1)*blen])
			vector = append(vector, v)
		}
		return NewColumnBinaryVector(fd.GetFieldName(), dim, vector), nil
	default:
		return nil, errors.New("unsupported data type")
	}
}
