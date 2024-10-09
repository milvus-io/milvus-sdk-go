package entity

import (
	"fmt"

	"github.com/cockroachdb/errors"
	"github.com/milvus-io/milvus-proto/go-api/v2/schemapb"
)

// ColumnVarCharArray generated columns type for VarChar
type ColumnVarCharArray struct {
	ColumnBase
	name        string
	values      [][][]byte
	validValues []bool
	nullable    bool
}

// Name returns column name
func (c *ColumnVarCharArray) Name() string {
	return c.name
}

// Type returns column FieldType
func (c *ColumnVarCharArray) Type() FieldType {
	return FieldTypeArray
}

// Len returns column values length
func (c *ColumnVarCharArray) Len() int {
	return len(c.values)
}

// Nullable returns column nullable
func (c *ColumnVarCharArray) Nullable() bool {
	return c.nullable
}

func (c *ColumnVarCharArray) Slice(start, end int) Column {
	l := c.Len()
	if start > l {
		start = l
	}
	if end == -1 || end > l {
		end = l
	}
	sliceValidValues := make([]bool, 0)
	if c.nullable {
		sliceValidValues = c.validValues[start:end]
	}
	return &ColumnVarCharArray{
		ColumnBase:  c.ColumnBase,
		name:        c.name,
		values:      c.values[start:end],
		validValues: sliceValidValues,
		nullable:    c.nullable,
	}
}

// Get returns value at index as interface{}.
func (c *ColumnVarCharArray) Get(idx int) (interface{}, error) {
	var r []string // use default value
	if idx < 0 || idx >= c.Len() {
		return r, errors.New("index out of range")
	}
	if c.nullable {
		if idx < 0 || idx >= len(c.validValues) {
			return nil, errors.New("index out of validValues range")
		}
		if !c.validValues[idx] {
			return nil, nil
		}
	}
	return c.values[idx], nil
}

// FieldData return column data mapped to schemapb.FieldData
func (c *ColumnVarCharArray) FieldData() *schemapb.FieldData {
	fd := &schemapb.FieldData{
		Type:      schemapb.DataType_Array,
		FieldName: c.name,
		ValidData: c.validValues,
	}
	convertTo := func(arr [][]byte) *schemapb.ScalarField {
		converted := make([]string, 0, c.Len())
		for i := 0; i < len(arr); i++ {
			converted = append(converted, string(arr[i]))
		}
		return &schemapb.ScalarField{
			Data: &schemapb.ScalarField_StringData{
				StringData: &schemapb.StringArray{
					Data: converted,
				},
			},
		}
	}

	data := make([]*schemapb.ScalarField, 0, len(c.validValues))
	if c.nullable {
		for i, v := range c.validValues {
			if v {
				data = append(data, convertTo(c.values[i]))
			}
		}
	} else {
		for _, arr := range c.values {
			data = append(data, convertTo(arr))
		}
	}

	fd.Field = &schemapb.FieldData_Scalars{
		Scalars: &schemapb.ScalarField{
			Data: &schemapb.ScalarField_ArrayData{
				ArrayData: &schemapb.ArrayArray{
					Data:        data,
					ElementType: schemapb.DataType_VarChar,
				},
			},
		},
	}
	return fd
}

// ValueByIdx returns value of the provided index
// error occurs when index out of range
func (c *ColumnVarCharArray) ValueByIdx(idx int) ([][]byte, error) {
	var r [][]byte // use default value
	if idx < 0 || idx >= c.Len() {
		return r, errors.New("index out of range")
	}
	if c.nullable {
		if idx < 0 || idx >= len(c.validValues) {
			return r, errors.New("index out of validValues range")
		}
		if !c.validValues[idx] {
			return nil, nil
		}
	}
	return c.values[idx], nil
}

// AppendValue append value into column
func (c *ColumnVarCharArray) AppendValue(i interface{}) error {
	var v [][]byte
	if i == nil && c.nullable {
		c.values = append(c.values, v)
		c.validValues = append(c.validValues, false)
		return nil
	}
	v, ok := i.([][]byte)
	if !ok {
		return fmt.Errorf("invalid type, expected []string, got %T", i)
	}
	c.values = append(c.values, v)
	if c.nullable {
		c.validValues = append(c.validValues, true)
	}

	return nil
}

// Data returns column data
func (c *ColumnVarCharArray) Data() [][][]byte {
	return c.values
}

// ValidData returns column validValues
func (c *ColumnVarCharArray) ValidData() []bool {
	return c.validValues
}

// NewColumnVarChar auto generated constructor
func NewColumnVarCharArray(name string, values [][][]byte) *ColumnVarCharArray {
	return &ColumnVarCharArray{
		name:   name,
		values: values,
	}
}

// NewNullableColumnJSONBytes composes a nullable Column with json bytes.
func NewNullableColumnVarCharArray(name string, values [][][]byte, validValues []bool) *ColumnVarCharArray {
	return &ColumnVarCharArray{
		name:        name,
		values:      values,
		nullable:    true,
		validValues: validValues,
	}
}
