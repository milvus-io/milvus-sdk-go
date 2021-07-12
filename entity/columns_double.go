// Code generated by go generate; DO NOT EDIT
// This file is generated by go genrated at 2021-07-12 12:48:54.240744206 &#43;0800 CST m=&#43;0.002833978

//Package entity defines entities used in sdk
package entity 

import "github.com/milvus-io/milvus-sdk-go/internal/proto/schema"

// ColumnDouble generated columns type for Double
type ColumnDouble struct {
	name   string
	values []float64
}

// Name returns column name
func (c *ColumnDouble) Name() string {
	return c.name
}

// Type returns column FieldType
func (c *ColumnDouble) Type() FieldType {
	return FieldTypeDouble
}

// Len returns column values length
func (c *ColumnDouble) Len() int {
	return len(c.values)
}

// FieldData return column data mapped to schema.FieldData
func (c *ColumnDouble) FieldData() *schema.FieldData {
	fd := &schema.FieldData{
		Type: schema.DataType_Double,
		FieldName: c.name,
	}
	fd.Field = &schema.FieldData_Scalars{
		Scalars: &schema.ScalarField{
			Data: &schema.ScalarField_DoubleData{
				DoubleData: &schema.DoubleArray{
					Data: []float64{},
				},
			},
		},
	}
	return fd
}

// NewColumnDouble auto generated constructor
func NewColumnDouble(name string, values []float64) *ColumnDouble {
	return &ColumnDouble {
		name: name,
		values: values,
	}
}
