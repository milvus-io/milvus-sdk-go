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
	"reflect"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/suite"
)

// ArrayRow test case type
type ArrayRow [16]float32

func (ar *ArrayRow) Collection() string  { return "" }
func (ar *ArrayRow) Partition() string   { return "" }
func (ar *ArrayRow) Description() string { return "" }

type Uint8Struct struct {
	RowBase
	Attr uint8
}

type StringArrayStruct struct {
	RowBase
	Vector [8]string
}

type StringSliceStruct struct {
	RowBase
	Vector []string `milvus:"dim:8"`
}

type SliceNoDimStruct struct {
	RowBase
	Vector []float32 `milvus:""`
}

type SliceBadDimStruct struct {
	RowBase
	Vector []float32 `milvus:"dim:str"`
}

type SliceBadDimStruct2 struct {
	RowBase
	Vector []float32 `milvus:"dim:0"`
}

func TestParseSchema(t *testing.T) {

	t.Run("invalid cases", func(t *testing.T) {
		// anonymous struct with default collection name ("") will cause error
		anonymusStruct := struct {
			RowBase
		}{}
		sch, err := ParseSchema(anonymusStruct)
		assert.Nil(t, sch)
		assert.NotNil(t, err)

		// MapRow
		m := make(MapRow)
		sch, err = ParseSchema(m)
		assert.Nil(t, sch)
		assert.NotNil(t, err)

		// non struct
		arrayRow := ArrayRow([16]float32{})
		sch, err = ParseSchema(&arrayRow)
		assert.Nil(t, sch)
		assert.NotNil(t, err)

		// uint8 not supported
		sch, err = ParseSchema(&Uint8Struct{})
		assert.Nil(t, sch)
		assert.NotNil(t, err)

		// string array not supported
		sch, err = ParseSchema(&StringArrayStruct{})
		assert.Nil(t, sch)
		assert.NotNil(t, err)

		// string slice not supported
		sch, err = ParseSchema(&StringSliceStruct{})
		assert.Nil(t, sch)
		assert.NotNil(t, err)

		// slice vector with no dim
		sch, err = ParseSchema(&SliceNoDimStruct{})
		assert.Nil(t, sch)
		assert.NotNil(t, err)

		// slice vector with bad format dim
		sch, err = ParseSchema(&SliceBadDimStruct{})
		assert.Nil(t, sch)
		assert.NotNil(t, err)

		// slice vector with bad format dim 2
		sch, err = ParseSchema(&SliceBadDimStruct2{})
		assert.Nil(t, sch)
		assert.NotNil(t, err)

	})

	t.Run("valid cases", func(t *testing.T) {

		sch, err := ParseSchema(RowBase{})
		assert.Nil(t, err)
		assert.Equal(t, "RowBase", sch.CollectionName)

		type ValidStruct struct {
			RowBase
			ID     int64 `milvus:"primary_key"`
			Attr1  int8
			Attr2  int16
			Attr3  int32
			Attr4  float32
			Attr5  float64
			Attr6  string
			Vector []float32 `milvus:"dim:128"`
		}
		vs := &ValidStruct{}
		sch, err = ParseSchema(vs)
		assert.Nil(t, err)
		assert.NotNil(t, sch)
		assert.Equal(t, "ValidStruct", sch.CollectionName)

		type ValidByteStruct struct {
			RowBase
			ID     int64  `milvus:"primary_key"`
			Vector []byte `milvus:"dim:128"`
		}
		vs2 := &ValidByteStruct{}
		sch, err = ParseSchema(vs2)
		assert.Nil(t, err)
		assert.NotNil(t, sch)

		type ValidArrayStruct struct {
			RowBase
			ID     int64 `milvus:"primary_key"`
			Vector [64]float32
		}
		vs3 := &ValidArrayStruct{}
		sch, err = ParseSchema(vs3)
		assert.Nil(t, err)
		assert.NotNil(t, sch)

		type ValidArrayStructByte struct {
			RowBase
			ID     int64   `milvus:"primary_key;auto_id"`
			Data   *string `milvus:"extra:test\\;false"`
			Vector [64]byte
		}
		vs4 := &ValidArrayStructByte{}
		sch, err = ParseSchema(vs4)
		assert.Nil(t, err)
		assert.NotNil(t, sch)

		vs5 := &ValidStructWithNamedTag{}
		sch, err = ParseSchema(vs5)
		assert.Nil(t, err)
		assert.NotNil(t, sch)
		i64f, vecf := false, false
		for _, field := range sch.Fields {
			if field.Name == "id" {
				i64f = true
			}
			if field.Name == "vector" {
				vecf = true
			}
		}

		assert.True(t, i64f)
		assert.True(t, vecf)
	})
}

type ValidStruct struct {
	RowBase
	ID      int64 `milvus:"primary_key"`
	Attr1   int8
	Attr2   int16
	Attr3   int32
	Attr4   float32
	Attr5   float64
	Attr6   string
	Attr7   bool
	Vector  []float32 `milvus:"dim:16"`
	Vector2 []byte    `milvus:"dim:32"`
}

type ValidStruct2 struct {
	RowBase
	ID      int64 `milvus:"primary_key"`
	Vector  [16]float32
	Vector2 [4]byte
	Ignored bool `milvus:"-"`
}

type ValidStructWithNamedTag struct {
	RowBase
	ID     int64       `milvus:"primary_key;name:id"`
	Vector [16]float32 `milvus:"name:vector"`
}

type RowsSuite struct {
	suite.Suite
}

func (s *RowsSuite) TestRowsToColumns() {
	s.Run("valid_cases", func() {

		columns, err := RowsToColumns([]Row{&ValidStruct{}})
		s.Nil(err)
		s.Equal(10, len(columns))

		columns, err = RowsToColumns([]Row{&ValidStruct2{}})
		s.Nil(err)
		s.Equal(3, len(columns))
	})

	s.Run("auto_id_pk", func() {
		type AutoPK struct {
			RowBase
			ID     int64     `milvus:"primary_key;auto_id"`
			Vector []float32 `milvus:"dim:32"`
		}
		columns, err := RowsToColumns([]Row{&AutoPK{}})
		s.Nil(err)
		s.Require().Equal(1, len(columns))
		s.Equal("Vector", columns[0].Name())
	})

	s.Run("invalid_cases", func() {
		// empty input
		_, err := RowsToColumns([]Row{})
		s.NotNil(err)

		// incompatible rows
		_, err = RowsToColumns([]Row{&ValidStruct{}, &ValidStruct2{}})
		s.NotNil(err)

		// schema & row not compatible
		_, err = RowsToColumns([]Row{&ValidStruct{}}, &Schema{
			Fields: []*Field{
				{
					Name:     "int64",
					DataType: FieldTypeInt64,
				},
			},
		})
		s.NotNil(err)
	})
}

func (s *RowsSuite) TestDynamicSchema() {
	s.Run("all_fallback_dynamic", func() {
		columns, err := RowsToColumns([]Row{&ValidStruct{}},
			NewSchema().WithDynamicFieldEnabled(true),
		)
		s.NoError(err)
		s.Equal(1, len(columns))
	})

	s.Run("dynamic_not_found", func() {
		_, err := RowsToColumns([]Row{&ValidStruct{}},
			NewSchema().WithField(
				NewField().WithName("ID").WithDataType(FieldTypeInt64).WithIsPrimaryKey(true),
			).WithDynamicFieldEnabled(true),
		)
		s.NoError(err)
	})
}

func (s *RowsSuite) TestReflectValueCandi() {
	cases := []struct {
		tag       string
		v         reflect.Value
		expect    map[string]fieldCandi
		expectErr bool
	}{
		{
			tag: "MapRow",
			v: reflect.ValueOf(MapRow(map[string]interface{}{
				"A": "abd", "B": int64(8),
			})),
			expect: map[string]fieldCandi{
				"A": {
					name: "A",
					v:    reflect.ValueOf("abd"),
				},
				"B": {
					name: "B",
					v:    reflect.ValueOf(int64(8)),
				},
			},
			expectErr: false,
		},
	}

	for _, c := range cases {
		s.Run(c.tag, func() {
			r, err := reflectValueCandi(c.v)
			if c.expectErr {
				s.Error(err)
				return
			}
			s.NoError(err)
			s.Equal(len(c.expect), len(r))
			for k, v := range c.expect {
				rv, has := r[k]
				s.Require().True(has)
				s.Equal(v.name, rv.name)
			}
		})
	}
}

func TestRows(t *testing.T) {
	suite.Run(t, new(RowsSuite))
}
