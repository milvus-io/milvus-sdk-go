// Code generated by go generate; DO NOT EDIT
// This file is generated by go genrated at 2021-07-12 12:48:54.240744206 &#43;0800 CST m=&#43;0.002833978

//Package entity defines entities used in sdk
package entity 

import "github.com/milvus-io/milvus-sdk-go/internal/proto/schema"


// ColumnFloatVector generated columns type for FloatVector
type ColumnFloatVector struct {
	name   string
	dim    int
	values [][]float32
}

// Name returns column name
func (c *ColumnFloatVector) Name() string {
	return c.name
}

// Type returns column FieldType
func (c *ColumnFloatVector) Type() FieldType {
	return FieldTypeFloatVector
}

// Len returns column data length
func (c * ColumnFloatVector) Len() int {
	return len(c.values)
}

// Dim returns vector dimension
func (c *ColumnFloatVector) Dim() int {
	return c.dim
}

// FieldData return column data mapped to schema.FieldData
func (c *ColumnFloatVector) FieldData() *schema.FieldData {
	fd := &schema.FieldData{
		Type: schema.DataType_FloatVector,
		FieldName: c.name,
	}

	data := make([]float32, 0, len(c.values)* c.dim)

	for _, vector := range c.values {
		data = append(data, vector...)
	}

	fd.Field = &schema.FieldData_Vectors{
		Vectors: &schema.VectorField{
			Dim: int64(c.dim),
			
			Data: &schema.VectorField_FloatVector{
				FloatVector: &schema.FloatArray{
					Data: data,
				},
			},
			
		},
	}
	return fd
}

// NewColumnFloatVector auto generated constructor
func NewColumnFloatVector(name string, dim int, values [][]float32) *ColumnFloatVector {
	return &ColumnFloatVector {
		name:   name,
		dim:    dim,
		values: values,
	}
}
