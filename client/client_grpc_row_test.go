package client

import (
	"context"
	"fmt"
	"reflect"
	"testing"

	"github.com/cockroachdb/errors"

	"github.com/golang/protobuf/proto"
	common "github.com/milvus-io/milvus-proto/go-api/commonpb"
	server "github.com/milvus-io/milvus-proto/go-api/milvuspb"
	schema "github.com/milvus-io/milvus-proto/go-api/schemapb"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
	"github.com/stretchr/testify/assert"
)

func TestCreateCollectionByRow(t *testing.T) {
	ctx := context.Background()
	c := testClient(ctx, t)
	type ValidStruct struct {
		entity.RowBase
		ID     int64 `milvus:"primary_key"`
		Attr1  int8
		Attr2  int16
		Attr3  int32
		Attr4  float32
		Attr5  float64
		Attr6  string
		Vector []float32 `milvus:"dim:4"`
	}
	t.Run("Test normal creation", func(t *testing.T) {
		mockServer.DelInjection(MHasCollection)
		shardsNum := int32(1)
		mockServer.SetInjection(MCreateCollection, func(ctx context.Context, raw proto.Message) (proto.Message, error) {
			req, ok := raw.(*server.CreateCollectionRequest)
			if !ok {
				return &common.Status{ErrorCode: common.ErrorCode_IllegalArgument}, errors.New("illegal request type")
			}
			assert.Equal(t, "ValidStruct", req.GetCollectionName())
			sschema := &schema.CollectionSchema{}
			if !assert.Nil(t, proto.Unmarshal(req.GetSchema(), sschema)) {
				assert.Equal(t, 8, len(sschema.Fields))
				assert.Equal(t, shardsNum, req.GetShardsNum())
			}

			return &common.Status{ErrorCode: common.ErrorCode_Success}, nil
		})
		assert.Nil(t, c.CreateCollectionByRow(ctx, &ValidStruct{}, shardsNum))
	})

	t.Run("Invalid cases", func(t *testing.T) {
		//Duplicated
		m := make(map[string]struct{})
		mockServer.SetInjection(MCreateCollection, func(_ context.Context, raw proto.Message) (proto.Message, error) {
			req, ok := raw.(*server.CreateCollectionRequest)
			if !ok {
				return BadRequestStatus()
			}
			m[req.GetCollectionName()] = struct{}{}

			return SuccessStatus()
		})
		mockServer.SetInjection(MHasCollection, func(_ context.Context, raw proto.Message) (proto.Message, error) {
			req, ok := raw.(*server.HasCollectionRequest)
			resp := &server.BoolResponse{}
			if !ok {
				return BadRequestStatus()
			}

			_, has := m[req.GetCollectionName()]
			resp.Value = has
			s, err := SuccessStatus()
			resp.Status = s
			return resp, err
		})
		assert.Nil(t, c.CreateCollectionByRow(ctx, &ValidStruct{}, 1))
		assert.NotNil(t, c.CreateCollectionByRow(ctx, &ValidStruct{}, 1))
		// Invalid struct
		anonymusStruct := struct {
			entity.RowBase
		}{}

		assert.NotNil(t, c.CreateCollectionByRow(ctx, &anonymusStruct, 1))
	})
}

func TestInsertByRows(t *testing.T) {
	ctx := context.Background()

	c := testClient(ctx, t)

	t.Run("test create failure due to meta", func(t *testing.T) {
		mockServer.DelInjection(MHasCollection) // collection does not exist
		ids, err := c.InsertByRows(ctx, testCollectionName, "", []entity.Row{})
		assert.Nil(t, ids)
		assert.NotNil(t, err)

		// partition not exists
		mockServer.SetInjection(MHasCollection, hasCollectionDefault)
		ids, err = c.InsertByRows(ctx, testCollectionName, "_part_not_exists", []entity.Row{})
		assert.Nil(t, ids)
		assert.NotNil(t, err)

		// field not in collection
		mockServer.SetInjection(MDescribeCollection, describeCollectionInjection(t, 0, testCollectionName, defaultSchema()))
		type ExtraFieldRow struct {
			entity.RowBase
			Int64      int64 `milvus:"primary_key"`
			ExtraField float64
			Vector     []float32 `milvus:"dim:128"`
		}
		ids, err = c.InsertByRows(ctx, testCollectionName, "", []entity.Row{&ExtraFieldRow{}})
		assert.Nil(t, ids)
		assert.NotNil(t, err)

		// field type not match
		mockServer.SetInjection(MDescribeCollection, describeCollectionInjection(t, 0, testCollectionName, defaultSchema()))
		type OtherFieldRow struct {
			entity.RowBase
			Int64  int32     `milvus:"primary_key;name:int64"`
			Vector []float32 `milvus:"dim:128;name:vector"`
		}
		ids, err = c.InsertByRows(ctx, testCollectionName, "", []entity.Row{
			&OtherFieldRow{},
		})
		assert.Nil(t, ids)
		assert.NotNil(t, err)

		// missing field
		type MissingFieldRow struct {
			entity.RowBase
			Vector []float32 `milvus:"dim:128;name:vector"`
		}
		ids, err = c.InsertByRows(ctx, testCollectionName, "", []entity.Row{
			&MissingFieldRow{},
		})
		assert.Nil(t, ids)
		assert.NotNil(t, err)
		t.Log(err.Error())

		// column len not match, not equvilant case in row based api

		// dim not match
		type Dim2Row struct {
			entity.RowBase
			Int64  int64     `milvus:"primary_key;name:int64"`
			Vector []float32 `milvus:"dim:16;name:vector"`
		}
		ids, err = c.InsertByRows(ctx, testCollectionName, "", []entity.Row{
			&Dim2Row{},
		})
		assert.Nil(t, ids)
		assert.NotNil(t, err)
		t.Log(err.Error())
	})

	mockServer.SetInjection(MHasCollection, hasCollectionDefault)
	mockServer.SetInjection(MDescribeCollection, describeCollectionInjection(t, 0, testCollectionName, defaultSchema()))

	vector := generateFloatVector(4096, testVectorDim)
	mockServer.SetInjection(MInsert, func(_ context.Context, raw proto.Message) (proto.Message, error) {
		req, ok := raw.(*server.InsertRequest)
		resp := &server.MutationResult{}
		if !ok {
			s, err := BadRequestStatus()
			resp.Status = s
			return resp, err
		}
		assert.EqualValues(t, 4096, req.GetNumRows())
		assert.Equal(t, testCollectionName, req.GetCollectionName())
		intIds := &schema.IDs_IntId{
			IntId: &schema.LongArray{
				Data: make([]int64, 4096),
			},
		}
		resp.IDs = &schema.IDs{
			IdField: intIds,
		}
		s, err := SuccessStatus()
		resp.Status = s
		return resp, err
	})
	_, err := c.Insert(ctx, testCollectionName, "", // use default partition
		entity.NewColumnFloatVector(testVectorField, testVectorDim, vector))

	assert.Nil(t, err)

}

func TestSearchResultToRows(t *testing.T) {
	t.Run("successful test cases", func(t *testing.T) {
		sr := &schema.SearchResultData{
			NumQueries: 1,
			TopK:       3,
			FieldsData: []*schema.FieldData{
				longFieldData("ID", []int64{1, 2, 3}),
				intFieldData("Attr1", []int32{1, 2, 3}),
				intFieldData("Attr2", []int32{1, 2, 3}),
				intFieldData("Attr3", []int32{1, 2, 3}),
				floatFieldData("Attr4", []float32{0.1, 0.2, 0.3}),
				doubleFieldData("Attr5", []float64{0.1, 0.2, 0.3}),
				stringFieldData("Attr6", []string{"1", "2", "3"}),
				floatVectorFieldData("Vector", 4, []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}),
			},
			Scores: []float32{0.1, 0.2, 0.3},
			Ids: &schema.IDs{
				IdField: &schema.IDs_IntId{
					IntId: &schema.LongArray{
						Data: []int64{1, 2, 3},
					},
				},
			},
			Topks: []int64{3},
		}

		type ValidStruct struct {
			entity.RowBase
			ID     int64 `milvus:"primary_key"`
			Attr1  int8
			Attr2  int16
			Attr3  int32
			Attr4  float32
			Attr5  float64
			Attr6  string
			Vector []float32 `milvus:"dim:4"`
		}
		sch, err := entity.ParseSchema(&ValidStruct{})
		assert.Nil(t, err)
		results, err := SearchResultToRows(sch, sr, reflect.TypeOf(&ValidStruct{}), arrMap("ID", "Vector", "Attr1", "Attr2", "Attr3", "Attr4", "Attr5", "Attr6"))
		assert.Nil(t, err)
		assert.NotNil(t, results)
		if assert.Equal(t, 1, len(results)) {
			result := results[0]
			assert.EqualValues(t, []float32{0.1, 0.2, 0.3}, result.Scores)
			if assert.Nil(t, result.Err) {
				if !assert.Equal(t, 3, len(result.Rows)) {
					t.FailNow()
				}
				for i, row := range result.Rows {
					s, ok := row.(*ValidStruct)
					if assert.True(t, ok) {
						assert.EqualValues(t, i+1, s.ID)
						assert.EqualValues(t, i+1, s.Attr1)
						assert.EqualValues(t, i+1, s.Attr2)
						assert.EqualValues(t, i+1, s.Attr3)
						assert.EqualValues(t, float32(i+1)/10.0, s.Attr4)
						assert.EqualValues(t, float64(i+1)/10.0, s.Attr5)
						assert.Equal(t, fmt.Sprintf("%d", i+1), s.Attr6)
					}
				}
			}
		}
	})
}

func arrMap(items ...string) map[string]struct{} {
	r := make(map[string]struct{})
	for _, item := range items {
		r[item] = struct{}{}
	}
	return r
}

func TestSetFieldValue(t *testing.T) {
	type TestStruct struct {
		Bool   bool
		Int8   int8
		Int16  int16
		Int32  int32
		Int64  int64
		Float  float32
		Double float64
		String string
		Arr    [8]float32
		ArrBin [8]byte
	}

	t.Run("successful cases", func(t *testing.T) {
		var err error
		item := &TestStruct{}
		// test bool field
		b := reflect.ValueOf(item).Elem().FieldByName("Bool")
		err = SetFieldValue(&entity.Field{
			DataType: entity.FieldTypeBool,
		}, b, boolFieldData("", []bool{true}), 0)
		assert.Nil(t, err)
		assert.EqualValues(t, true, item.Bool)

		// test int 8 field
		i8 := reflect.ValueOf(item).Elem().FieldByName("Int8")
		err = SetFieldValue(&entity.Field{
			DataType: entity.FieldTypeInt8,
		}, i8, intFieldData("", []int32{10}), 0)
		assert.Nil(t, err)
		assert.EqualValues(t, 10, item.Int8)

		i16 := reflect.ValueOf(item).Elem().FieldByName("Int16")
		err = SetFieldValue(&entity.Field{
			DataType: entity.FieldTypeInt16,
		}, i16, intFieldData("", []int32{10}), 0)
		assert.Nil(t, err)
		assert.EqualValues(t, 10, item.Int16)

		i32 := reflect.ValueOf(item).Elem().FieldByName("Int32")
		err = SetFieldValue(&entity.Field{
			DataType: entity.FieldTypeInt32,
		}, i32, intFieldData("", []int32{10}), 0)
		assert.Nil(t, err)
		assert.EqualValues(t, 10, item.Int32)

		i64 := reflect.ValueOf(item).Elem().FieldByName("Int64")
		err = SetFieldValue(&entity.Field{
			DataType: entity.FieldTypeInt64,
		}, i64, longFieldData("", []int64{10}), 0)
		assert.Nil(t, err)
		assert.EqualValues(t, 10, item.Int64)

		f32 := reflect.ValueOf(item).Elem().FieldByName("Float")
		err = SetFieldValue(&entity.Field{
			DataType: entity.FieldTypeFloat,
		}, f32, floatFieldData("", []float32{0.618}), 0)
		assert.Nil(t, err)
		assert.InDelta(t, 0.618, item.Float, 1e-6)

		f64 := reflect.ValueOf(item).Elem().FieldByName("Double")
		err = SetFieldValue(&entity.Field{
			DataType: entity.FieldTypeDouble,
		}, f64, doubleFieldData("", []float64{0.618}), 0)
		assert.Nil(t, err)
		assert.InDelta(t, 0.618, item.Double, 1e-6)

		str := reflect.ValueOf(item).Elem().FieldByName("String")
		err = SetFieldValue(&entity.Field{
			DataType: entity.FieldTypeString,
		}, str, stringFieldData("", []string{"test"}), 0)
		assert.Nil(t, err)
		assert.EqualValues(t, "test", item.String)

		// float32 array field
		arr := reflect.ValueOf(item).Elem().FieldByName("Arr")
		err = SetFieldValue(&entity.Field{
			DataType: entity.FieldTypeFloatVector,
		}, arr, floatVectorFieldData("", 8, []float32{0, 1, 2, 3, 4, 5, 6, 7}), 0)
		assert.Nil(t, err)
		assert.EqualValues(t, [8]float32{0, 1, 2, 3, 4, 5, 6, 7}, item.Arr)

		binArr := reflect.ValueOf(item).Elem().FieldByName("ArrBin")
		err = SetFieldValue(&entity.Field{
			DataType: entity.FieldTypeBinaryVector,
		}, binArr, binaryVectorFieldData("", []byte{'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a'}), 0)
		assert.Nil(t, err)
		assert.EqualValues(t, [8]byte{'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a'}, item.ArrBin)
	})

	t.Run("fail cases", func(t *testing.T) {
		var err error
		item := &TestStruct{}
		// invalid data type
		b := reflect.ValueOf(item).Elem().FieldByName("Bool")
		i8 := reflect.ValueOf(item).Elem().FieldByName("Int8")
		i16 := reflect.ValueOf(item).Elem().FieldByName("Int16")
		i32 := reflect.ValueOf(item).Elem().FieldByName("Int32")
		i64 := reflect.ValueOf(item).Elem().FieldByName("Int64")
		f32 := reflect.ValueOf(item).Elem().FieldByName("Float")
		f64 := reflect.ValueOf(item).Elem().FieldByName("Double")
		str := reflect.ValueOf(item).Elem().FieldByName("String")
		vf := reflect.ValueOf(item).Elem().FieldByName("Arr")
		//vb := reflect.ValueOf(item).Elem().FieldByName("ArrBin")

		err = SetFieldValue(&entity.Field{
			DataType: entity.FieldTypeNone,
		}, b, boolFieldData("", []bool{true}), 0)
		assert.Equal(t, ErrFieldTypeNotMatch, err)
		// field type not matched cases

		// bool
		err = SetFieldValue(&entity.Field{DataType: entity.FieldTypeBool}, i8, boolFieldData("", []bool{true}), 0)
		assert.Equal(t, err, ErrFieldTypeNotMatch)
		err = SetFieldValue(&entity.Field{DataType: entity.FieldTypeBool}, b, emptyFieldData(), 0)
		assert.Equal(t, err, ErrFieldTypeNotMatch)
		err = SetFieldValue(&entity.Field{DataType: entity.FieldTypeBool}, b, emptyScalarFieldData(), 0)
		assert.Equal(t, err, ErrFieldTypeNotMatch)

		// int8
		err = SetFieldValue(&entity.Field{DataType: entity.FieldTypeInt8}, b, intFieldData("", []int32{10}), 0)
		assert.Equal(t, err, ErrFieldTypeNotMatch)
		err = SetFieldValue(&entity.Field{DataType: entity.FieldTypeInt8}, i8, emptyFieldData(), 0)
		assert.Equal(t, err, ErrFieldTypeNotMatch)
		err = SetFieldValue(&entity.Field{DataType: entity.FieldTypeInt8}, i8, emptyScalarFieldData(), 0)
		assert.Equal(t, err, ErrFieldTypeNotMatch)

		// int16
		err = SetFieldValue(&entity.Field{DataType: entity.FieldTypeInt16}, b, intFieldData("", []int32{10}), 0)
		assert.Equal(t, err, ErrFieldTypeNotMatch)
		err = SetFieldValue(&entity.Field{DataType: entity.FieldTypeInt16}, i16, emptyFieldData(), 0)
		assert.Equal(t, err, ErrFieldTypeNotMatch)
		err = SetFieldValue(&entity.Field{DataType: entity.FieldTypeInt16}, i16, emptyScalarFieldData(), 0)
		assert.Equal(t, err, ErrFieldTypeNotMatch)

		// int32
		err = SetFieldValue(&entity.Field{DataType: entity.FieldTypeInt32}, b, intFieldData("", []int32{10}), 0)
		assert.Equal(t, err, ErrFieldTypeNotMatch)
		err = SetFieldValue(&entity.Field{DataType: entity.FieldTypeInt32}, i32, emptyFieldData(), 0)
		assert.Equal(t, err, ErrFieldTypeNotMatch)
		err = SetFieldValue(&entity.Field{DataType: entity.FieldTypeInt32}, i32, emptyScalarFieldData(), 0)
		assert.Equal(t, err, ErrFieldTypeNotMatch)

		// int64
		err = SetFieldValue(&entity.Field{DataType: entity.FieldTypeInt64}, b, longFieldData("", []int64{10}), 0)
		assert.Equal(t, err, ErrFieldTypeNotMatch)
		err = SetFieldValue(&entity.Field{DataType: entity.FieldTypeInt64}, i64, emptyFieldData(), 0)
		assert.Equal(t, err, ErrFieldTypeNotMatch)
		err = SetFieldValue(&entity.Field{DataType: entity.FieldTypeInt64}, i64, emptyScalarFieldData(), 0)
		assert.Equal(t, err, ErrFieldTypeNotMatch)

		// float
		err = SetFieldValue(&entity.Field{DataType: entity.FieldTypeFloat}, b, floatFieldData("", []float32{0.6}), 0)
		assert.Equal(t, err, ErrFieldTypeNotMatch)
		err = SetFieldValue(&entity.Field{DataType: entity.FieldTypeFloat}, f32, emptyFieldData(), 0)
		assert.Equal(t, err, ErrFieldTypeNotMatch)
		err = SetFieldValue(&entity.Field{DataType: entity.FieldTypeFloat}, f32, emptyScalarFieldData(), 0)
		assert.Equal(t, err, ErrFieldTypeNotMatch)

		// double
		err = SetFieldValue(&entity.Field{DataType: entity.FieldTypeDouble}, b, doubleFieldData("", []float64{0.6}), 0)
		assert.Equal(t, err, ErrFieldTypeNotMatch)
		err = SetFieldValue(&entity.Field{DataType: entity.FieldTypeDouble}, f64, emptyFieldData(), 0)
		assert.Equal(t, err, ErrFieldTypeNotMatch)
		err = SetFieldValue(&entity.Field{DataType: entity.FieldTypeDouble}, f64, emptyScalarFieldData(), 0)
		assert.Equal(t, err, ErrFieldTypeNotMatch)

		// string
		err = SetFieldValue(&entity.Field{DataType: entity.FieldTypeString}, b, stringFieldData("", []string{"test"}), 0)
		assert.Equal(t, err, ErrFieldTypeNotMatch)
		err = SetFieldValue(&entity.Field{DataType: entity.FieldTypeString}, str, emptyFieldData(), 0)
		assert.Equal(t, err, ErrFieldTypeNotMatch)
		err = SetFieldValue(&entity.Field{DataType: entity.FieldTypeString}, str, emptyScalarFieldData(), 0)
		assert.Equal(t, err, ErrFieldTypeNotMatch)

		// float vector
		err = SetFieldValue(&entity.Field{DataType: entity.FieldTypeFloatVector}, b, floatVectorFieldData("", 4, []float32{1, 2, 3, 4}), 0)
		assert.Equal(t, err, ErrFieldTypeNotMatch)
		err = SetFieldValue(&entity.Field{DataType: entity.FieldTypeFloatVector}, vf, emptyFieldData(), 0)
		assert.Equal(t, err, ErrFieldTypeNotMatch)
		err = SetFieldValue(&entity.Field{DataType: entity.FieldTypeFloatVector}, vf, emptyVectorFieldData(), 0)
		assert.Equal(t, err, ErrFieldTypeNotMatch)

		// binary vector
		err = SetFieldValue(&entity.Field{DataType: entity.FieldTypeBinaryVector}, b, binaryVectorFieldData("", []byte{1, 2, 3, 4}), 0)
		assert.Equal(t, err, ErrFieldTypeNotMatch)
		err = SetFieldValue(&entity.Field{DataType: entity.FieldTypeBinaryVector}, vf, emptyFieldData(), 0)
		assert.Equal(t, err, ErrFieldTypeNotMatch)
		err = SetFieldValue(&entity.Field{DataType: entity.FieldTypeBinaryVector}, vf, emptyVectorFieldData(), 0)
		assert.Equal(t, err, ErrFieldTypeNotMatch)
	})
}

func emptyFieldData() *schema.FieldData {
	return &schema.FieldData{}
}
func emptyScalarFieldData() *schema.FieldData {
	return &schema.FieldData{
		Field: &schema.FieldData_Scalars{
			Scalars: &schema.ScalarField{},
		},
	}
}

func emptyVectorFieldData() *schema.FieldData {
	return &schema.FieldData{
		Field: &schema.FieldData_Vectors{
			Vectors: &schema.VectorField{},
		},
	}
}

func boolFieldData(name string, data []bool) *schema.FieldData {
	return &schema.FieldData{
		FieldName: name,
		Type:      schema.DataType_Bool,
		Field: &schema.FieldData_Scalars{
			Scalars: &schema.ScalarField{
				Data: &schema.ScalarField_BoolData{
					BoolData: &schema.BoolArray{
						Data: data,
					},
				},
			},
		},
	}
}

func intFieldData(name string, data []int32) *schema.FieldData {
	return &schema.FieldData{
		FieldName: name,
		// Type not determined
		Field: &schema.FieldData_Scalars{
			Scalars: &schema.ScalarField{
				Data: &schema.ScalarField_IntData{
					IntData: &schema.IntArray{
						Data: data,
					},
				},
			},
		},
	}
}

func longFieldData(name string, data []int64) *schema.FieldData {
	return &schema.FieldData{
		FieldName: name,
		Type:      schema.DataType_Int64,
		Field: &schema.FieldData_Scalars{
			Scalars: &schema.ScalarField{
				Data: &schema.ScalarField_LongData{
					LongData: &schema.LongArray{
						Data: data,
					},
				},
			},
		},
	}
}

func floatFieldData(name string, data []float32) *schema.FieldData {
	return &schema.FieldData{
		FieldName: name,
		Type:      schema.DataType_Float,
		Field: &schema.FieldData_Scalars{
			Scalars: &schema.ScalarField{
				Data: &schema.ScalarField_FloatData{
					FloatData: &schema.FloatArray{
						Data: data,
					},
				},
			},
		},
	}
}

func doubleFieldData(name string, data []float64) *schema.FieldData {
	return &schema.FieldData{
		FieldName: name,
		Type:      schema.DataType_Double,
		Field: &schema.FieldData_Scalars{
			Scalars: &schema.ScalarField{
				Data: &schema.ScalarField_DoubleData{
					DoubleData: &schema.DoubleArray{
						Data: data,
					},
				},
			},
		},
	}
}

func stringFieldData(name string, data []string) *schema.FieldData {
	return &schema.FieldData{
		FieldName: name,
		Type:      schema.DataType_String,
		Field: &schema.FieldData_Scalars{
			Scalars: &schema.ScalarField{
				Data: &schema.ScalarField_StringData{
					StringData: &schema.StringArray{
						Data: data,
					},
				},
			},
		},
	}
}

func floatVectorFieldData(name string, dim int, data []float32) *schema.FieldData {
	return &schema.FieldData{
		FieldName: name,
		Type:      schema.DataType_FloatVector,
		Field: &schema.FieldData_Vectors{
			Vectors: &schema.VectorField{
				Dim: int64(dim),
				Data: &schema.VectorField_FloatVector{
					FloatVector: &schema.FloatArray{
						Data: data,
					},
				},
			},
		},
	}
}

func binaryVectorFieldData(name string, data []byte) *schema.FieldData {
	return &schema.FieldData{
		FieldName: name,
		Type:      schema.DataType_BinaryVector,
		Field: &schema.FieldData_Vectors{
			Vectors: &schema.VectorField{
				Dim: int64(8 * len(data)),
				Data: &schema.VectorField_BinaryVector{
					BinaryVector: data,
				},
			},
		},
	}
}
