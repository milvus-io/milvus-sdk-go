package entity

import (
	"math/rand"
	"strconv"
	"testing"
	"time"

	schema "github.com/milvus-io/milvus-proto/go-api/schemapb"
	"github.com/stretchr/testify/assert"
)

func TestVectors(t *testing.T) {
	rand.Seed(time.Now().UnixNano())
	dim := rand.Intn(127) + 1

	t.Run("test float vector", func(t *testing.T) {
		raw := make([]float32, dim)
		for i := 0; i < dim; i++ {
			raw[i] = rand.Float32()
		}

		fv := FloatVector(raw)

		assert.Equal(t, dim, fv.Dim())
		assert.Equal(t, dim*4, len(fv.Serialize()))
	})

	t.Run("test binary vector", func(t *testing.T) {
		raw := make([]byte, dim)
		_, err := rand.Read(raw)
		assert.Nil(t, err)

		bv := BinaryVector(raw)

		assert.Equal(t, dim*8, bv.Dim())
		assert.ElementsMatch(t, raw, bv.Serialize())
	})
}

func TestIDColumns(t *testing.T) {
	dataLen := rand.Intn(100) + 1
	base := rand.Intn(5000) // id start point

	t.Run("nil id", func(t *testing.T) {
		_, err := IDColumns(nil, 0, -1)
		assert.NotNil(t, err)
		idField := &schema.IDs{}
		_, err = IDColumns(idField, 0, -1)
		assert.NotNil(t, err)
	})

	t.Run("int ids", func(t *testing.T) {
		ids := make([]int64, 0, dataLen)
		for i := 0; i < dataLen; i++ {
			ids = append(ids, int64(i+base))
		}
		idField := &schema.IDs{
			IdField: &schema.IDs_IntId{
				IntId: &schema.LongArray{
					Data: ids,
				},
			},
		}
		column, err := IDColumns(idField, 0, dataLen)
		assert.Nil(t, err)
		assert.NotNil(t, column)
		assert.Equal(t, dataLen, column.Len())

		column, err = IDColumns(idField, 0, -1) // test -1 method
		assert.Nil(t, err)
		assert.NotNil(t, column)
		assert.Equal(t, dataLen, column.Len())
	})
	t.Run("string ids", func(t *testing.T) {
		ids := make([]string, 0, dataLen)
		for i := 0; i < dataLen; i++ {
			ids = append(ids, strconv.FormatInt(int64(i+base), 10))
		}
		idField := &schema.IDs{
			IdField: &schema.IDs_StrId{
				StrId: &schema.StringArray{
					Data: ids,
				},
			},
		}
		column, err := IDColumns(idField, 0, dataLen)
		assert.Nil(t, err)
		assert.NotNil(t, column)
		assert.Equal(t, dataLen, column.Len())

		column, err = IDColumns(idField, 0, -1) // test -1 method
		assert.Nil(t, err)
		assert.NotNil(t, column)
		assert.Equal(t, dataLen, column.Len())
	})
}

func TestGetIntData(t *testing.T) {
	type testCase struct {
		tag      string
		fd       *schema.FieldData
		expectOK bool
	}

	cases := []testCase{
		{
			tag: "normal_IntData",
			fd: &schema.FieldData{
				Field: &schema.FieldData_Scalars{
					Scalars: &schema.ScalarField{
						Data: &schema.ScalarField_IntData{
							IntData: &schema.IntArray{Data: []int32{1, 2, 3}},
						},
					},
				},
			},
			expectOK: true,
		},
		{
			tag: "empty_LongData",
			fd: &schema.FieldData{
				Field: &schema.FieldData_Scalars{
					Scalars: &schema.ScalarField{
						Data: &schema.ScalarField_LongData{
							LongData: &schema.LongArray{Data: nil},
						},
					},
				},
			},
			expectOK: true,
		},
		{
			tag: "nonempty_LongData",
			fd: &schema.FieldData{
				Field: &schema.FieldData_Scalars{
					Scalars: &schema.ScalarField{
						Data: &schema.ScalarField_LongData{
							LongData: &schema.LongArray{Data: []int64{1, 2, 3}},
						},
					},
				},
			},
			expectOK: false,
		},
		{
			tag: "other_data",
			fd: &schema.FieldData{
				Field: &schema.FieldData_Scalars{
					Scalars: &schema.ScalarField{
						Data: &schema.ScalarField_BoolData{},
					},
				},
			},
			expectOK: false,
		},
		{
			tag: "vector_data",
			fd: &schema.FieldData{
				Field: &schema.FieldData_Vectors{},
			},
			expectOK: false,
		},
	}

	for _, tc := range cases {
		t.Run(tc.tag, func(t *testing.T) {
			_, ok := getIntData(tc.fd)
			assert.Equal(t, tc.expectOK, ok)
		})
	}
}
