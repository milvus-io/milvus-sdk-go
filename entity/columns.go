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

	"github.com/milvus-io/milvus-proto/go-api/v2/schemapb"

	"github.com/cockroachdb/errors"
)

//go:generate go run gen/gen.go

// Column interface field type for column-based data frame
type Column interface {
	Name() string
	Type() FieldType
	Len() int
	Nullable() bool
	Slice(int, int) Column
	FieldData() *schemapb.FieldData
	AppendValue(interface{}) error
	Get(int) (interface{}, error)
	GetAsInt64(int) (int64, error)
	GetAsString(int) (string, error)
	GetAsDouble(int) (float64, error)
	GetAsBool(int) (bool, error)
}

// ColumnBase adds conversion methods support for fixed-type columns.
type ColumnBase struct{}

func (b ColumnBase) GetAsInt64(_ int) (int64, error) {
	return 0, errors.New("conversion between fixed-type column not support")
}

func (b ColumnBase) GetAsString(_ int) (string, error) {
	return "", errors.New("conversion between fixed-type column not support")
}

func (b ColumnBase) GetAsDouble(_ int) (float64, error) {
	return 0, errors.New("conversion between fixed-type column not support")
}

func (b ColumnBase) GetAsBool(_ int) (bool, error) {
	return false, errors.New("conversion between fixed-type column not support")
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

// FloatVector float32 vector wrapper.
type Float16Vector []byte

// Dim returns vector dimension.
func (fv Float16Vector) Dim() int {
	return len(fv) / 2
}

// FieldType returns coresponding field type.
func (fv Float16Vector) FieldType() FieldType {
	return FieldTypeFloat16Vector
}

func (fv Float16Vector) Serialize() []byte {
	return fv
}

// FloatVector float32 vector wrapper.
type BFloat16Vector []byte

// Dim returns vector dimension.
func (fv BFloat16Vector) Dim() int {
	return len(fv) / 2
}

// FieldType returns coresponding field type.
func (fv BFloat16Vector) FieldType() FieldType {
	return FieldTypeBFloat16Vector
}

func (fv BFloat16Vector) Serialize() []byte {
	return fv
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

// IDColumns converts schemapb.IDs to corresponding column
// currently Int64 / string may be in IDs
func IDColumns(schema *Schema, idField *schemapb.IDs, begin, end int) (Column, error) {
	var idColumn Column

	pkField := schema.PKField()
	if pkField == nil {
		return nil, errors.New("PK Field not found")
	}
	switch pkField.DataType {
	case FieldTypeInt64:
		data := idField.GetIntId().GetData()
		if data == nil {
			return NewColumnInt64(pkField.Name, nil), nil
		}
		if end >= 0 {
			idColumn = NewColumnInt64(pkField.Name, data[begin:end])
		} else {
			idColumn = NewColumnInt64(pkField.Name, data[begin:])
		}
	case FieldTypeVarChar, FieldTypeString:
		data := idField.GetStrId().GetData()
		if data == nil {
			return NewColumnVarChar(pkField.Name, nil), nil
		}
		if end >= 0 {
			idColumn = NewColumnVarChar(pkField.Name, data[begin:end])
		} else {
			idColumn = NewColumnVarChar(pkField.Name, data[begin:])
		}
	default:
		return nil, fmt.Errorf("unsupported id type %v", pkField.DataType)
	}
	if idField == nil {
		return nil, errors.New("nil Ids from response")
	}
	return idColumn, nil
}

// FieldDataColumn converts schemapb.FieldData to Column, used int search result conversion logic
// begin, end specifies the start and end positions
func FieldDataColumn(fd *schemapb.FieldData, begin, end int) (Column, error) {
	switch fd.GetType() {
	case schemapb.DataType_Bool:
		data, ok := fd.GetScalars().GetData().(*schemapb.ScalarField_BoolData)
		if !ok {
			return nil, errFieldDataTypeNotMatch
		}
		if len(fd.GetValidData()) != 0 {
			if end < 0 {
				return NewNullableColumnBool(fd.GetFieldName(), data.BoolData.GetData()[begin:], fd.GetValidData()[begin:]), nil
			}
			return NewNullableColumnBool(fd.GetFieldName(), data.BoolData.GetData()[begin:end], fd.GetValidData()[begin:end]), nil
		}

		if end < 0 {
			return NewColumnBool(fd.GetFieldName(), data.BoolData.GetData()[begin:]), nil
		}
		return NewColumnBool(fd.GetFieldName(), data.BoolData.GetData()[begin:end]), nil

	case schemapb.DataType_Int8:
		data, ok := getIntData(fd)
		if !ok {
			return nil, errFieldDataTypeNotMatch
		}
		values := make([]int8, 0, len(data.IntData.GetData()))
		for _, v := range data.IntData.GetData() {
			values = append(values, int8(v))
		}

		if len(fd.GetValidData()) != 0 {
			if end < 0 {
				return NewNullableColumnInt8(fd.GetFieldName(), values[begin:], fd.GetValidData()[begin:]), nil
			}
			return NewNullableColumnInt8(fd.GetFieldName(), values[begin:end], fd.GetValidData()[begin:end]), nil
		}

		if end < 0 {
			return NewColumnInt8(fd.GetFieldName(), values[begin:]), nil
		}

		return NewColumnInt8(fd.GetFieldName(), values[begin:end]), nil

	case schemapb.DataType_Int16:
		data, ok := getIntData(fd)
		if !ok {
			return nil, errFieldDataTypeNotMatch
		}
		values := make([]int16, 0, len(data.IntData.GetData()))
		for _, v := range data.IntData.GetData() {
			values = append(values, int16(v))
		}
		if len(fd.GetValidData()) != 0 {
			if end < 0 {
				return NewNullableColumnInt16(fd.GetFieldName(), values[begin:], fd.GetValidData()[begin:]), nil
			}
			return NewNullableColumnInt16(fd.GetFieldName(), values[begin:end], fd.GetValidData()[begin:end]), nil
		}

		if end < 0 {
			return NewColumnInt16(fd.GetFieldName(), values[begin:]), nil
		}
		return NewColumnInt16(fd.GetFieldName(), values[begin:end]), nil

	case schemapb.DataType_Int32:
		data, ok := getIntData(fd)
		if !ok {
			return nil, errFieldDataTypeNotMatch
		}
		if len(fd.GetValidData()) != 0 {
			if end < 0 {
				return NewNullableColumnInt32(fd.GetFieldName(), data.IntData.GetData()[begin:], fd.GetValidData()[begin:]), nil
			}
			return NewNullableColumnInt32(fd.GetFieldName(), data.IntData.GetData()[begin:end], fd.GetValidData()[begin:end]), nil
		}
		if end < 0 {
			return NewColumnInt32(fd.GetFieldName(), data.IntData.GetData()[begin:]), nil
		}
		return NewColumnInt32(fd.GetFieldName(), data.IntData.GetData()[begin:end]), nil

	case schemapb.DataType_Int64:
		data, ok := fd.GetScalars().GetData().(*schemapb.ScalarField_LongData)
		if !ok {
			return nil, errFieldDataTypeNotMatch
		}
		if len(fd.GetValidData()) != 0 {
			if end < 0 {
				return NewNullableColumnInt64(fd.GetFieldName(), data.LongData.GetData()[begin:], fd.GetValidData()[begin:]), nil
			}
			return NewNullableColumnInt64(fd.GetFieldName(), data.LongData.GetData()[begin:end], fd.GetValidData()[begin:end]), nil
		}
		if end < 0 {
			return NewColumnInt64(fd.GetFieldName(), data.LongData.GetData()[begin:]), nil
		}
		return NewColumnInt64(fd.GetFieldName(), data.LongData.GetData()[begin:end]), nil

	case schemapb.DataType_Float:
		data, ok := fd.GetScalars().GetData().(*schemapb.ScalarField_FloatData)
		if !ok {
			return nil, errFieldDataTypeNotMatch
		}
		if len(fd.GetValidData()) != 0 {
			if end < 0 {
				return NewNullableColumnFloat(fd.GetFieldName(), data.FloatData.GetData()[begin:], fd.GetValidData()[begin:]), nil
			}
			return NewNullableColumnFloat(fd.GetFieldName(), data.FloatData.GetData()[begin:end], fd.GetValidData()[begin:end]), nil
		}
		if end < 0 {
			return NewColumnFloat(fd.GetFieldName(), data.FloatData.GetData()[begin:]), nil
		}
		return NewColumnFloat(fd.GetFieldName(), data.FloatData.GetData()[begin:end]), nil

	case schemapb.DataType_Double:
		data, ok := fd.GetScalars().GetData().(*schemapb.ScalarField_DoubleData)
		if !ok {
			return nil, errFieldDataTypeNotMatch
		}
		if len(fd.GetValidData()) != 0 {
			if end < 0 {
				return NewNullableColumnDouble(fd.GetFieldName(), data.DoubleData.GetData()[begin:], fd.GetValidData()[begin:]), nil
			}
			return NewNullableColumnDouble(fd.GetFieldName(), data.DoubleData.GetData()[begin:end], fd.GetValidData()[begin:end]), nil
		}
		if end < 0 {
			return NewColumnDouble(fd.GetFieldName(), data.DoubleData.GetData()[begin:]), nil
		}
		return NewColumnDouble(fd.GetFieldName(), data.DoubleData.GetData()[begin:end]), nil

	case schemapb.DataType_String:
		data, ok := fd.GetScalars().GetData().(*schemapb.ScalarField_StringData)
		if !ok {
			return nil, errFieldDataTypeNotMatch
		}
		if len(fd.GetValidData()) != 0 {
			if end < 0 {
				return NewNullableColumnString(fd.GetFieldName(), data.StringData.GetData()[begin:], fd.GetValidData()[begin:]), nil
			}
			return NewNullableColumnString(fd.GetFieldName(), data.StringData.GetData()[begin:end], fd.GetValidData()[begin:end]), nil
		}
		if end < 0 {
			return NewColumnString(fd.GetFieldName(), data.StringData.GetData()[begin:]), nil
		}
		return NewColumnString(fd.GetFieldName(), data.StringData.GetData()[begin:end]), nil

	case schemapb.DataType_VarChar:
		data, ok := fd.GetScalars().GetData().(*schemapb.ScalarField_StringData)
		if !ok {
			return nil, errFieldDataTypeNotMatch
		}
		if len(fd.GetValidData()) != 0 {
			if end < 0 {
				return NewNullableColumnVarChar(fd.GetFieldName(), data.StringData.GetData()[begin:], fd.GetValidData()[begin:]), nil
			}
			return NewNullableColumnVarChar(fd.GetFieldName(), data.StringData.GetData()[begin:end], fd.GetValidData()[begin:end]), nil
		}
		if end < 0 {
			return NewColumnVarChar(fd.GetFieldName(), data.StringData.GetData()[begin:]), nil
		}
		return NewColumnVarChar(fd.GetFieldName(), data.StringData.GetData()[begin:end]), nil

	case schemapb.DataType_Array:
		data := fd.GetScalars().GetArrayData()
		if data == nil {
			return nil, errFieldDataTypeNotMatch
		}
		var arrayData []*schemapb.ScalarField
		if end < 0 {
			arrayData = data.GetData()[begin:]
		} else {
			arrayData = data.GetData()[begin:end]
		}

		if len(fd.GetValidData()) != 0 {
			if end < 0 {
				return parseNullableArrayData(fd.GetFieldName(), data.GetElementType(), arrayData, fd.GetValidData()[begin:])
			}
			return parseNullableArrayData(fd.GetFieldName(), data.GetElementType(), arrayData, fd.GetValidData()[begin:end])
		}

		return parseArrayData(fd.GetFieldName(), data.GetElementType(), arrayData)

	case schemapb.DataType_JSON:
		data, ok := fd.GetScalars().GetData().(*schemapb.ScalarField_JsonData)
		isDynamic := fd.GetIsDynamic()
		if !ok {
			return nil, errFieldDataTypeNotMatch
		}
		if len(fd.GetValidData()) != 0 {
			if end < 0 {
				return NewNullableColumnJSONBytes(fd.GetFieldName(), data.JsonData.GetData()[begin:], fd.GetValidData()[begin:]).WithIsDynamic(isDynamic), nil
			}
			return NewNullableColumnJSONBytes(fd.GetFieldName(), data.JsonData.GetData()[begin:end], fd.GetValidData()[begin:end]).WithIsDynamic(isDynamic), nil
		}
		if end < 0 {
			return NewColumnJSONBytes(fd.GetFieldName(), data.JsonData.GetData()[begin:]).WithIsDynamic(isDynamic), nil
		}
		return NewColumnJSONBytes(fd.GetFieldName(), data.JsonData.GetData()[begin:end]).WithIsDynamic(isDynamic), nil

	case schemapb.DataType_FloatVector:
		vectors := fd.GetVectors()
		x, ok := vectors.GetData().(*schemapb.VectorField_FloatVector)
		if !ok {
			return nil, errFieldDataTypeNotMatch
		}
		data := x.FloatVector.GetData()
		dim := int(vectors.GetDim())
		if end < 0 {
			end = int(len(data) / dim)
		}
		vector := make([][]float32, 0, end-begin) // shall not have remanunt
		for i := begin; i < end; i++ {
			v := make([]float32, dim)
			copy(v, data[i*dim:(i+1)*dim])
			vector = append(vector, v)
		}
		return NewColumnFloatVector(fd.GetFieldName(), dim, vector), nil

	case schemapb.DataType_BinaryVector:
		vectors := fd.GetVectors()
		x, ok := vectors.GetData().(*schemapb.VectorField_BinaryVector)
		if !ok {
			return nil, errFieldDataTypeNotMatch
		}
		data := x.BinaryVector
		if data == nil {
			return nil, errFieldDataTypeNotMatch
		}
		dim := int(vectors.GetDim())
		blen := dim / 8
		if end < 0 {
			end = int(len(data) / blen)
		}
		vector := make([][]byte, 0, end-begin)
		for i := begin; i < end; i++ {
			v := make([]byte, blen)
			copy(v, data[i*blen:(i+1)*blen])
			vector = append(vector, v)
		}
		return NewColumnBinaryVector(fd.GetFieldName(), dim, vector), nil

	case schemapb.DataType_Float16Vector:
		vectors := fd.GetVectors()
		x, ok := vectors.GetData().(*schemapb.VectorField_Float16Vector)
		if !ok {
			return nil, errFieldDataTypeNotMatch
		}
		data := x.Float16Vector
		dim := int(vectors.GetDim())
		if end < 0 {
			end = int(len(data) / dim / 2)
		}
		vector := make([][]byte, 0, end-begin)
		for i := begin; i < end; i++ {
			v := make([]byte, dim*2)
			copy(v, data[i*dim*2:(i+1)*dim*2])
			vector = append(vector, v)
		}
		return NewColumnFloat16Vector(fd.GetFieldName(), dim, vector), nil

	case schemapb.DataType_BFloat16Vector:
		vectors := fd.GetVectors()
		x, ok := vectors.GetData().(*schemapb.VectorField_Bfloat16Vector)
		if !ok {
			return nil, errFieldDataTypeNotMatch
		}
		data := x.Bfloat16Vector
		dim := int(vectors.GetDim())

		if end < 0 {
			end = int(len(data) / dim / 2)
		}
		vector := make([][]byte, 0, end-begin) // shall not have remanunt
		for i := begin; i < end; i++ {
			v := make([]byte, dim*2)
			copy(v, data[i*dim*2:(i+1)*dim*2])
			vector = append(vector, v)
		}
		return NewColumnBFloat16Vector(fd.GetFieldName(), dim, vector), nil
	case schemapb.DataType_SparseFloatVector:
		sparseVectors := fd.GetVectors().GetSparseFloatVector()
		if sparseVectors == nil {
			return nil, errFieldDataTypeNotMatch
		}
		data := sparseVectors.Contents
		if end < 0 {
			end = len(data)
		}
		data = data[begin:end]
		vectors := make([]SparseEmbedding, 0, len(data))
		for _, bs := range data {
			vector, err := deserializeSliceSparceEmbedding(bs)
			if err != nil {
				return nil, err
			}
			vectors = append(vectors, vector)
		}
		return NewColumnSparseVectors(fd.GetFieldName(), vectors), nil
	default:
		return nil, fmt.Errorf("unsupported data type %s", fd.GetType())
	}
}

func parseNullableArrayData(fieldName string, elementType schemapb.DataType, fieldDataList []*schemapb.ScalarField, validData []bool) (Column, error) {
	switch elementType {
	case schemapb.DataType_Bool:
		var data [][]bool
		for _, fd := range fieldDataList {
			data = append(data, fd.GetBoolData().GetData())
		}
		return NewNullableColumnBoolArray(fieldName, data, validData), nil

	case schemapb.DataType_Int8:
		var data [][]int8
		for _, fd := range fieldDataList {
			raw := fd.GetIntData().GetData()
			row := make([]int8, 0, len(raw))
			for _, item := range raw {
				row = append(row, int8(item))
			}
			data = append(data, row)
		}
		return NewNullableColumnInt8Array(fieldName, data, validData), nil

	case schemapb.DataType_Int16:
		var data [][]int16
		for _, fd := range fieldDataList {
			raw := fd.GetIntData().GetData()
			row := make([]int16, 0, len(raw))
			for _, item := range raw {
				row = append(row, int16(item))
			}
			data = append(data, row)
		}
		return NewNullableColumnInt16Array(fieldName, data, validData), nil

	case schemapb.DataType_Int32:
		var data [][]int32
		for _, fd := range fieldDataList {
			data = append(data, fd.GetIntData().GetData())
		}
		return NewNullableColumnInt32Array(fieldName, data, validData), nil

	case schemapb.DataType_Int64:
		var data [][]int64
		for _, fd := range fieldDataList {
			data = append(data, fd.GetLongData().GetData())
		}
		return NewNullableColumnInt64Array(fieldName, data, validData), nil

	case schemapb.DataType_Float:
		var data [][]float32
		for _, fd := range fieldDataList {
			data = append(data, fd.GetFloatData().GetData())
		}
		return NewNullableColumnFloatArray(fieldName, data, validData), nil

	case schemapb.DataType_Double:
		var data [][]float64
		for _, fd := range fieldDataList {
			data = append(data, fd.GetDoubleData().GetData())
		}
		return NewNullableColumnDoubleArray(fieldName, data, validData), nil

	case schemapb.DataType_VarChar, schemapb.DataType_String:
		var data [][][]byte
		for _, fd := range fieldDataList {
			strs := fd.GetStringData().GetData()
			bytesData := make([][]byte, 0, len(strs))
			for _, str := range strs {
				bytesData = append(bytesData, []byte(str))
			}
			data = append(data, bytesData)
		}

		return NewNullableColumnVarCharArray(fieldName, data, validData), nil

	default:
		return nil, fmt.Errorf("unsupported element type %s", elementType)
	}
}

func parseArrayData(fieldName string, elementType schemapb.DataType, fieldDataList []*schemapb.ScalarField) (Column, error) {

	switch elementType {
	case schemapb.DataType_Bool:
		var data [][]bool
		for _, fd := range fieldDataList {
			data = append(data, fd.GetBoolData().GetData())
		}
		return NewColumnBoolArray(fieldName, data), nil

	case schemapb.DataType_Int8:
		var data [][]int8
		for _, fd := range fieldDataList {
			raw := fd.GetIntData().GetData()
			row := make([]int8, 0, len(raw))
			for _, item := range raw {
				row = append(row, int8(item))
			}
			data = append(data, row)
		}
		return NewColumnInt8Array(fieldName, data), nil

	case schemapb.DataType_Int16:
		var data [][]int16
		for _, fd := range fieldDataList {
			raw := fd.GetIntData().GetData()
			row := make([]int16, 0, len(raw))
			for _, item := range raw {
				row = append(row, int16(item))
			}
			data = append(data, row)
		}
		return NewColumnInt16Array(fieldName, data), nil

	case schemapb.DataType_Int32:
		var data [][]int32
		for _, fd := range fieldDataList {
			data = append(data, fd.GetIntData().GetData())
		}
		return NewColumnInt32Array(fieldName, data), nil

	case schemapb.DataType_Int64:
		var data [][]int64
		for _, fd := range fieldDataList {
			data = append(data, fd.GetLongData().GetData())
		}
		return NewColumnInt64Array(fieldName, data), nil

	case schemapb.DataType_Float:
		var data [][]float32
		for _, fd := range fieldDataList {
			data = append(data, fd.GetFloatData().GetData())
		}
		return NewColumnFloatArray(fieldName, data), nil

	case schemapb.DataType_Double:
		var data [][]float64
		for _, fd := range fieldDataList {
			data = append(data, fd.GetDoubleData().GetData())
		}
		return NewColumnDoubleArray(fieldName, data), nil

	case schemapb.DataType_VarChar, schemapb.DataType_String:
		var data [][][]byte
		for _, fd := range fieldDataList {
			strs := fd.GetStringData().GetData()
			bytesData := make([][]byte, 0, len(strs))
			for _, str := range strs {
				bytesData = append(bytesData, []byte(str))
			}
			data = append(data, bytesData)
		}

		return NewColumnVarCharArray(fieldName, data), nil

	default:
		return nil, fmt.Errorf("unsupported element type %s", elementType)
	}
}

// getIntData get int32 slice from result field data
// also handles LongData bug (see also https://github.com/milvus-io/milvus/issues/23850)
func getIntData(fd *schemapb.FieldData) (*schemapb.ScalarField_IntData, bool) {
	switch data := fd.GetScalars().GetData().(type) {
	case *schemapb.ScalarField_IntData:
		return data, true
	case *schemapb.ScalarField_LongData:
		// only alway empty LongData for backward compatibility
		if len(data.LongData.GetData()) == 0 {
			return &schemapb.ScalarField_IntData{
				IntData: &schemapb.IntArray{},
			}, true
		}
		return nil, false
	default:
		return nil, false
	}
}

// FieldDataColumn converts schemapb.FieldData to vector Column
func FieldDataVector(fd *schemapb.FieldData) (Column, error) {
	switch fd.GetType() {
	case schemapb.DataType_FloatVector:
		vectors := fd.GetVectors()
		x, ok := vectors.GetData().(*schemapb.VectorField_FloatVector)
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
	case schemapb.DataType_BinaryVector:
		vectors := fd.GetVectors()
		x, ok := vectors.GetData().(*schemapb.VectorField_BinaryVector)
		if !ok {
			return nil, errFieldDataTypeNotMatch
		}
		data := x.BinaryVector
		if data == nil {
			return nil, errFieldDataTypeNotMatch
		}
		dim := int(vectors.GetDim())
		blen := dim / 8
		vector := make([][]byte, 0, len(data)/blen)
		for i := 0; i < len(data)/blen; i++ {
			v := make([]byte, blen)
			copy(v, data[i*blen:(i+1)*blen])
			vector = append(vector, v)
		}
		return NewColumnBinaryVector(fd.GetFieldName(), dim, vector), nil
	case schemapb.DataType_Float16Vector:
		vectors := fd.GetVectors()
		x, ok := vectors.GetData().(*schemapb.VectorField_Float16Vector)
		if !ok {
			return nil, errFieldDataTypeNotMatch
		}
		data := x.Float16Vector
		dim := int(vectors.GetDim())
		vector := make([][]byte, 0, len(data)/dim) // shall not have remanunt
		for i := 0; i < len(data)/dim; i++ {
			v := make([]byte, dim)
			copy(v, data[i*dim:(i+1)*dim])
			vector = append(vector, v)
		}
		return NewColumnFloat16Vector(fd.GetFieldName(), dim, vector), nil
	case schemapb.DataType_BFloat16Vector:
		vectors := fd.GetVectors()
		x, ok := vectors.GetData().(*schemapb.VectorField_Bfloat16Vector)
		if !ok {
			return nil, errFieldDataTypeNotMatch
		}
		data := x.Bfloat16Vector
		dim := int(vectors.GetDim())
		vector := make([][]byte, 0, len(data)/dim) // shall not have remanunt
		for i := 0; i < len(data)/dim; i++ {
			v := make([]byte, dim)
			copy(v, data[i*dim:(i+1)*dim])
			vector = append(vector, v)
		}
		return NewColumnBFloat16Vector(fd.GetFieldName(), dim, vector), nil
	default:
		return nil, errors.New("unsupported data type")
	}
}

// defaultValueColumn will return the empty scalars column which will be fill with default value
func DefaultValueColumn(name string, dataType FieldType) (Column, error) {
	switch dataType {
	case FieldTypeBool:
		return NewColumnBool(name, nil), nil
	case FieldTypeInt8:
		return NewColumnInt8(name, nil), nil
	case FieldTypeInt16:
		return NewColumnInt16(name, nil), nil
	case FieldTypeInt32:
		return NewColumnInt32(name, nil), nil
	case FieldTypeInt64:
		return NewColumnInt64(name, nil), nil
	case FieldTypeFloat:
		return NewColumnFloat(name, nil), nil
	case FieldTypeDouble:
		return NewColumnDouble(name, nil), nil
	case FieldTypeString:
		return NewColumnString(name, nil), nil
	case FieldTypeVarChar:
		return NewColumnVarChar(name, nil), nil
	case FieldTypeJSON:
		return NewColumnJSONBytes(name, nil), nil

	default:
		return nil, fmt.Errorf("default value unsupported data type %s", dataType)
	}
}

// NewAllNullValueColumn will return the empty scalars with nullable==true, and fill it with null
func NewAllNullValueColumn(name string, dataType FieldType, rowSize int) (Column, error) {
	switch dataType {
	case FieldTypeBool:
		return NewAllNullColumnBool(name, rowSize), nil
	case FieldTypeInt8:
		return NewAllNullColumnInt8(name, rowSize), nil
	case FieldTypeInt16:
		return NewAllNullColumnInt16(name, rowSize), nil
	case FieldTypeInt32:
		return NewAllNullColumnInt32(name, rowSize), nil
	case FieldTypeInt64:
		return NewAllNullColumnInt64(name, rowSize), nil
	case FieldTypeFloat:
		return NewAllNullColumnFloat(name, rowSize), nil
	case FieldTypeDouble:
		return NewAllNullColumnDouble(name, rowSize), nil
	case FieldTypeString:
		return NewAllNullColumnString(name, rowSize), nil
	case FieldTypeVarChar:
		return NewAllNullColumnVarChar(name, rowSize), nil
	case FieldTypeJSON:
		return NewAllNullColumnJSONBytes(name, rowSize), nil

	default:
		return nil, fmt.Errorf("default value unsupported data type %s", dataType)
	}
}
