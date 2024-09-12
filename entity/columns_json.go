package entity

import (
	"encoding/json"
	"fmt"
	"reflect"

	"github.com/cockroachdb/errors"
	schema "github.com/milvus-io/milvus-proto/go-api/v2/schemapb"
)

var _ (Column) = (*ColumnJSONBytes)(nil)

// ColumnJSONBytes column type for JSON.
// all items are marshaled json bytes.
type ColumnJSONBytes struct {
	ColumnBase
	name        string
	values      [][]byte
	isDynamic   bool
	validValues []bool
	nullable    bool
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

// Nullable returns column nullable
func (c *ColumnJSONBytes) Nullable() bool {
	return c.nullable
}

func (c *ColumnJSONBytes) Slice(start, end int) Column {
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
	return &ColumnJSONBytes{
		ColumnBase:  c.ColumnBase,
		name:        c.name,
		values:      c.values[start:end],
		validValues: sliceValidValues,
		nullable:    c.nullable,
	}
}

// Get returns value at index as interface{}.
func (c *ColumnJSONBytes) Get(idx int) (interface{}, error) {
	if idx < 0 || idx > c.Len() {
		return nil, errors.New("index out of range")
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

func (c *ColumnJSONBytes) GetAsString(idx int) (string, error) {
	bs, err := c.ValueByIdx(idx)
	if err != nil {
		return "", err
	}
	return string(bs), nil
}

// FieldData return column data mapped to schema.FieldData.
func (c *ColumnJSONBytes) FieldData() *schema.FieldData {
	fd := &schema.FieldData{
		Type:      schema.DataType_JSON,
		FieldName: c.name,
		IsDynamic: c.isDynamic,
		ValidData: c.validValues,
	}

	data := make([][]byte, 0, c.Len())
	if c.nullable {
		for i, v := range c.validValues {
			if v {
				data = append(data, c.values[i])
			}
		}
	} else {
		data = c.values
	}

	fd.Field = &schema.FieldData_Scalars{
		Scalars: &schema.ScalarField{
			Data: &schema.ScalarField_JsonData{
				JsonData: &schema.JSONArray{
					Data: data,
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

// AppendValue append value into column.
func (c *ColumnJSONBytes) AppendValue(i interface{}) error {
	var v []byte
	if i == nil && c.nullable {
		c.values = append(c.values, v)
		c.validValues = append(c.validValues, false)
		return nil
	}

	switch raw := i.(type) {
	case []byte:
		v = raw
	default:
		k := reflect.TypeOf(i).Kind()
		if k == reflect.Ptr {
			k = reflect.TypeOf(i).Elem().Kind()
		}
		switch k {
		case reflect.Struct:
			fallthrough
		case reflect.Map:
			bs, err := json.Marshal(raw)
			if err != nil {
				return err
			}
			v = bs
		default:
			return fmt.Errorf("expect json compatible type([]byte, struct[}, map], got %T)", i)
		}
	}
	c.values = append(c.values, v)
	if c.nullable {
		c.validValues = append(c.validValues, true)
	}

	return nil
}

// Data returns column data.
func (c *ColumnJSONBytes) Data() [][]byte {
	return c.values
}

// ValidData returns column validValues
func (c *ColumnJSONBytes) ValidData() []bool {
	return c.validValues
}

func (c *ColumnJSONBytes) WithIsDynamic(isDynamic bool) *ColumnJSONBytes {
	c.isDynamic = isDynamic
	return c
}

func (c *ColumnJSONBytes) IsDynamic() bool {
	return c.isDynamic
}

// NewColumnJSONBytes composes a Column with json bytes.
func NewColumnJSONBytes(name string, values [][]byte) *ColumnJSONBytes {
	return &ColumnJSONBytes{
		name:   name,
		values: values,
	}
}

// NewNullableColumnJSONBytes composes a nullable Column with json bytes.
func NewNullableColumnJSONBytes(name string, values [][]byte, validValues []bool) *ColumnJSONBytes {
	return &ColumnJSONBytes{
		name:        name,
		values:      values,
		nullable:    true,
		validValues: validValues,
	}
}

// NewAllNullColumnJSONBytes composes a nullable Column with json bytes.
func NewAllNullColumnJSONBytes(name string, rowSize int) *ColumnJSONBytes {
	return &ColumnJSONBytes{
		name:        name,
		values:      make([][]byte, rowSize),
		nullable:    true,
		validValues: make([]bool, rowSize),
	}
}
