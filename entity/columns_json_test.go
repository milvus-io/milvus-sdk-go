package entity

import (
	"fmt"
	"math/rand"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"
)

type ColumnJSONBytesSuite struct {
	suite.Suite
}

func (s *ColumnJSONBytesSuite) SetupSuite() {
	rand.Seed(time.Now().UnixNano())
}

func (s *ColumnJSONBytesSuite) TestAttrMethods() {
	columnName := fmt.Sprintf("column_jsonbs_%d", rand.Int())
	columnLen := 8 + rand.Intn(10)

	v := make([][]byte, columnLen)
	column := NewColumnJSONBytes(columnName, v).WithIsDynamic(true)

	s.Run("test_meta", func() {
		ft := FieldTypeJSON
		s.Equal("JSON", ft.Name())
		s.Equal("JSON", ft.String())
		pbName, pbType := ft.PbFieldType()
		s.Equal("JSON", pbName)
		s.Equal("JSON", pbType)
	})

	s.Run("test_column_attribute", func() {
		s.Equal(columnName, column.Name())
		s.Equal(FieldTypeJSON, column.Type())
		s.Equal(columnLen, column.Len())
		s.EqualValues(v, column.Data())
	})

	s.Run("test_column_field_data", func() {
		fd := column.FieldData()
		s.NotNil(fd)
		s.Equal(fd.GetFieldName(), columnName)
	})

	s.Run("test_column_valuer_by_idx", func() {
		_, err := column.ValueByIdx(-1)
		s.Error(err)
		_, err = column.ValueByIdx(columnLen)
		s.Error(err)
		for i := 0; i < columnLen; i++ {
			v, err := column.ValueByIdx(i)
			s.NoError(err)
			s.Equal(column.values[i], v)
		}
	})

	s.Run("test_append_value", func() {
		item := make([]byte, 10)
		err := column.AppendValue(item)
		s.NoError(err)
		s.Equal(columnLen+1, column.Len())
		val, err := column.ValueByIdx(columnLen)
		s.NoError(err)
		s.Equal(item, val)

		err = column.AppendValue(1)
		s.Error(err)
	})
}

func TestColumnJSONBytes(t *testing.T) {
	suite.Run(t, new(ColumnJSONBytesSuite))
}
