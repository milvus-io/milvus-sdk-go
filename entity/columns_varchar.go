package entity

import (
	"errors"
	"fmt"

	schema "github.com/milvus-io/milvus-proto/go-api/v2/schemapb"
)

// ColumnVarChar generated columns type for VarChar
type ColumnVarChar struct {
	ColumnBase
	name        string
	values      []string
	validValues []bool
	nullable    bool
}

// Name returns column name
func (c *ColumnVarChar) Name() string {
	return c.name
}

// Type returns column FieldType
func (c *ColumnVarChar) Type() FieldType {
	return FieldTypeVarChar
}

// Len returns column values length
func (c *ColumnVarChar) Len() int {
	return len(c.values)
}

// Nullable returns column nullable
func (c *ColumnVarChar) Nullable() bool {
	return c.nullable
}

// Get returns value at index as interface{}.
func (c *ColumnVarChar) Get(idx int) (interface{}, error) {
	if idx < 0 || idx > c.Len() {
		return "", errors.New("index out of range")
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

// GetAsString returns value at idx.
func (c *ColumnVarChar) GetAsString(idx int) (string, error) {
	if c.nullable {
		if idx < 0 || idx >= len(c.validValues) {
			return "", errors.New("index out of validValues range")
		}
		if !c.validValues[idx] {
			return "", nil
		}
	}
	if idx < 0 || idx > c.Len() {
		return "", errors.New("index out of range")
	}
	return c.values[idx], nil
}

func (c *ColumnVarChar) Slice(start, end int) Column {
	l := c.Len()
	if start > l {
		start = l
	}
	if end == -1 || end > l {
		end = l
	}
	sliceValidValues := make([]bool, 0, end-start)
	if c.nullable {
		sliceValidValues = c.validValues[start:end]
	}
	return &ColumnVarChar{
		ColumnBase:  c.ColumnBase,
		name:        c.name,
		values:      c.values[start:end],
		validValues: sliceValidValues,
		nullable:    c.nullable,
	}
}

// FieldData return column data mapped to schema.FieldData
func (c *ColumnVarChar) FieldData() *schema.FieldData {
	fd := &schema.FieldData{
		Type:      schema.DataType_VarChar,
		FieldName: c.name,
		ValidData: c.validValues,
	}
	data := make([]string, 0, c.Len())
	if c.nullable {
		for i, v := range c.validValues {
			if v {
				data = append(data, string(c.values[i]))
			}
		}
	} else {
		for i := 0; i < c.Len(); i++ {
			data = append(data, string(c.values[i]))
		}
	}

	fd.Field = &schema.FieldData_Scalars{
		Scalars: &schema.ScalarField{
			Data: &schema.ScalarField_StringData{
				StringData: &schema.StringArray{
					Data: data,
				},
			},
		},
	}
	return fd
}

// ValueByIdx returns value of the provided index
// error occurs when index out of range
func (c *ColumnVarChar) ValueByIdx(idx int) (string, error) {
	var r string // use default value
	if idx < 0 || idx >= c.Len() {
		return r, errors.New("index out of range")
	}
	if c.nullable {
		if idx < 0 || idx >= len(c.validValues) {
			return r, errors.New("index out of validValues range")
		}
		if !c.validValues[idx] {
			return r, nil
		}
	}
	return c.values[idx], nil
}

// AppendValue append value into column
func (c *ColumnVarChar) AppendValue(i interface{}) error {
	v, ok := i.(string)
	if !ok {
		return fmt.Errorf("invalid type, expected string, got %T", i)
	}
	c.values = append(c.values, v)

	return nil
}

// Data returns column data
func (c *ColumnVarChar) Data() []string {
	return c.values
}

// NewColumnVarChar auto generated constructor
func NewColumnVarChar(name string, values []string) *ColumnVarChar {
	return &ColumnVarChar{
		name:   name,
		values: values,
	}
}

// NewColumnVarChar auto generated constructor
func NewNullableColumnVarChar(name string, values []string, validValues []bool) *ColumnVarChar {
	return &ColumnVarChar{
		name:        name,
		values:      values,
		validValues: validValues,
		nullable:    true,
	}
}

// NewAllNullColumnVarChar auto generated constructor
func NewAllNullColumnVarChar(name string, rowSize int) *ColumnVarChar {
	return &ColumnVarChar{
		name:        name,
		values:      make([]string, rowSize),
		nullable:    true,
		validValues: make([]bool, rowSize),
	}
}
