package entity

import (
	"errors"
	"fmt"

	schema "github.com/milvus-io/milvus-proto/go-api/schemapb"
)

// ColumnVarChar generated columns type for VarChar
type ColumnVarChar struct {
	name   string
	values []string
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

// FieldData return column data mapped to schema.FieldData
func (c *ColumnVarChar) FieldData() *schema.FieldData {
	fd := &schema.FieldData{
		Type:      schema.DataType_VarChar,
		FieldName: c.name,
	}
	data := make([]string, 0, c.Len())
	for i := 0; i < c.Len(); i++ {
		data = append(data, string(c.values[i]))
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
