package entity

import (
	"testing"

	"github.com/stretchr/testify/assert"
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
		for _, f := range sch.Fields {
			t.Logf("%#v", f)
		}

		type ValidByteStruct struct {
			RowBase
			ID     int64  `milvus:"primary_key"`
			Vector []byte `milvus:"dim:128"`
		}
		vs2 := &ValidByteStruct{}
		sch, err = ParseSchema(vs2)
		assert.Nil(t, err)
		assert.NotNil(t, sch)
		for _, f := range sch.Fields {
			t.Logf("%#v", f)
		}

		type ValidArrayStruct struct {
			RowBase
			ID     int64 `milvus:"primary_key"`
			Vector [64]float32
		}
		vs3 := &ValidArrayStruct{}
		sch, err = ParseSchema(vs3)
		assert.Nil(t, err)
		assert.NotNil(t, sch)
		for _, f := range sch.Fields {
			t.Logf("%#v", f)
		}

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
		for _, f := range sch.Fields {
			t.Logf("%#v", f)
		}
	})
}
