package entity

import (
	"fmt"
	"math/rand"
	"testing"
	"time"

	schema "github.com/milvus-io/milvus-proto/go-api/schemapb"
	"github.com/stretchr/testify/assert"
)

func TestColumnVarChar(t *testing.T) {
	rand.Seed(time.Now().UnixNano())
	columnName := fmt.Sprintf("column_VarChar_%d", rand.Int())
	columnLen := 8 + rand.Intn(10)

	v := make([]string, columnLen)
	column := NewColumnVarChar(columnName, v)

	t.Run("test meta", func(t *testing.T) {
		ft := FieldTypeVarChar
		assert.Equal(t, "VarChar", ft.Name())
		assert.Equal(t, "string", ft.String())
		pbName, pbType := ft.PbFieldType()
		assert.Equal(t, "VarChar", pbName)
		assert.Equal(t, "string", pbType)
	})

	t.Run("test column attribute", func(t *testing.T) {
		assert.Equal(t, columnName, column.Name())
		assert.Equal(t, FieldTypeVarChar, column.Type())
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
		assert.NotNil(t, err)
		_, err = column.ValueByIdx(columnLen)
		assert.NotNil(t, err)
		for i := 0; i < columnLen; i++ {
			v, err := column.ValueByIdx(i)
			assert.Nil(t, err)
			assert.Equal(t, column.values[i], v)
		}
	})
}

func TestFieldDataVarCharColumn(t *testing.T) {
	colLen := rand.Intn(10) + 8
	name := fmt.Sprintf("fd_VarChar_%d", rand.Int())
	fd := &schema.FieldData{
		Type:      schema.DataType_VarChar,
		FieldName: name,
	}

	t.Run("normal usage", func(t *testing.T) {
		fd.Field = &schema.FieldData_Scalars{
			Scalars: &schema.ScalarField{
				Data: &schema.ScalarField_StringData{
					StringData: &schema.StringArray{
						Data: make([]string, colLen),
					},
				},
			},
		}
		column, err := FieldDataColumn(fd, 0, colLen)
		assert.Nil(t, err)
		assert.NotNil(t, column)

		assert.Equal(t, name, column.Name())
		assert.Equal(t, colLen, column.Len())
		assert.Equal(t, FieldTypeVarChar, column.Type())

		var ev string
		err = column.AppendValue(ev)
		assert.Equal(t, colLen+1, column.Len())
		assert.Nil(t, err)

		err = column.AppendValue(struct{}{})
		assert.Equal(t, colLen+1, column.Len())
		assert.NotNil(t, err)
	})

	t.Run("nil data", func(t *testing.T) {
		fd.Field = nil
		_, err := FieldDataColumn(fd, 0, colLen)
		assert.NotNil(t, err)
	})

	t.Run("get all data", func(t *testing.T) {
		fd.Field = &schema.FieldData_Scalars{
			Scalars: &schema.ScalarField{
				Data: &schema.ScalarField_StringData{
					StringData: &schema.StringArray{
						Data: make([]string, colLen),
					},
				},
			},
		}
		column, err := FieldDataColumn(fd, 0, -1)
		assert.Nil(t, err)
		assert.NotNil(t, column)

		assert.Equal(t, name, column.Name())
		assert.Equal(t, colLen, column.Len())
		assert.Equal(t, FieldTypeVarChar, column.Type())
	})
}
