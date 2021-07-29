package client

import (
	"reflect"
	"testing"

	"github.com/milvus-io/milvus-sdk-go/v2/entity"
	"github.com/milvus-io/milvus-sdk-go/v2/internal/proto/schema"
	"github.com/stretchr/testify/assert"
)

func TestSetFieldValue(t *testing.T) {
	type TestStruct struct {
		Int8  int8
		Int16 int16
		Int32 int32
		Int64 int64
		Arr   [8]float32
	}

	var err error
	item := &TestStruct{}
	i8 := reflect.ValueOf(item).Elem().FieldByName("Int8")
	err = SetFieldValue(&entity.Field{
		DataType: entity.FieldTypeInt8,
	}, i8, &schema.FieldData{
		Field: &schema.FieldData_Scalars{
			Scalars: &schema.ScalarField{
				Data: &schema.ScalarField_IntData{
					IntData: &schema.IntArray{
						Data: []int32{10},
					},
				},
			},
		},
	}, 0)
	assert.Nil(t, err)
	assert.EqualValues(t, 10, item.Int8)

	i16 := reflect.ValueOf(item).Elem().FieldByName("Int16")
	err = SetFieldValue(&entity.Field{
		DataType: entity.FieldTypeInt16,
	}, i16, &schema.FieldData{
		Field: &schema.FieldData_Scalars{
			Scalars: &schema.ScalarField{
				Data: &schema.ScalarField_IntData{
					IntData: &schema.IntArray{
						Data: []int32{10},
					},
				},
			},
		},
	}, 0)
	assert.Nil(t, err)
	assert.EqualValues(t, 10, item.Int16)

	i32 := reflect.ValueOf(item).Elem().FieldByName("Int32")
	err = SetFieldValue(&entity.Field{
		DataType: entity.FieldTypeInt32,
	}, i32, &schema.FieldData{
		Field: &schema.FieldData_Scalars{
			Scalars: &schema.ScalarField{
				Data: &schema.ScalarField_IntData{
					IntData: &schema.IntArray{
						Data: []int32{10},
					},
				},
			},
		},
	}, 0)
	assert.Nil(t, err)
	assert.EqualValues(t, 10, item.Int32)

	arr := reflect.ValueOf(item).Elem().FieldByName("Arr")
	err = SetFieldValue(&entity.Field{
		DataType: entity.FieldTypeFloatVector,
	}, arr, &schema.FieldData{
		Field: &schema.FieldData_Vectors{
			Vectors: &schema.VectorField{
				Dim: 8,
				Data: &schema.VectorField_FloatVector{
					FloatVector: &schema.FloatArray{
						Data: []float32{0, 1, 2, 3, 4, 5, 6, 7},
					},
				},
			},
		},
	}, 0)
	assert.Nil(t, err)
	assert.EqualValues(t, [8]float32{0, 1, 2, 3, 4, 5, 6, 7}, item.Arr)
}
