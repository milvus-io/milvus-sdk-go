package entity

import (
	"fmt"

	"github.com/cockroachdb/errors"
	schema "github.com/milvus-io/milvus-proto/go-api/schemapb"
)

var _ (Column) = (*ColumnJSONBytes)(nil)

// ColumnJSONBytes column type for JSON.
// all items are marshaled json bytes.
type ColumnJSONBytes struct {
	name   string
	values [][]byte
}

// Name returns column name.
func (c *ColumnJSONBytes) Name() string {
	return c.name
}

// Type returns column FieldType.
func (c *ColumnJSONBytes) Type() FieldType {
	return FieldTypeJSON
}

// Len returns column values length.
func (c *ColumnJSONBytes) Len() int {
	return len(c.values)
}

// Get returns value at index as interface{}.
func (c *ColumnJSONBytes) Get(idx int) (interface{}, error) {
	if idx < 0 || idx > c.Len() {
		return nil, errors.New("index out of range")
	}
	return c.values[idx], nil
}

// FieldData return column data mapped to schema.FieldData.
func (c *ColumnJSONBytes) FieldData() *schema.FieldData {
	fd := &schema.FieldData{
		Type:      schema.DataType_JSON,
		FieldName: c.name,
	}

	fd.Field = &schema.FieldData_Scalars{
		Scalars: &schema.ScalarField{
			Data: &schema.ScalarField_JsonData{
				JsonData: &schema.JSONArray{
					Data: c.values,
				},
			},
		},
	}

	return fd
}

// ValueByIdx returns value of the provided index.
func (c *ColumnJSONBytes) ValueByIdx(idx int) ([]byte, error) {
	if idx < 0 || idx >= c.Len() {
		return nil, errors.New("index out of range")
	}
	return c.values[idx], nil
}

// AppendValue append value into column.
func (c *ColumnJSONBytes) AppendValue(i interface{}) error {
	v, ok := i.([]byte)
	if !ok {
		return fmt.Errorf("invalid type, expected []byte, got %T", i)
	}
	c.values = append(c.values, v)

	return nil
}

// Data returns column data.
func (c *ColumnJSONBytes) Data() [][]byte {
	return c.values
}

// NewColumnJSONBytes composes a Column with json bytes.
func NewColumnJSONBytes(name string, values [][]byte) *ColumnJSONBytes {
	return &ColumnJSONBytes{
		name:   name,
		values: values,
	}
}
