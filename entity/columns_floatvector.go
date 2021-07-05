// Code generated by go generate; DO NOT EDIT
// This file is generated by go genrated at 2021-07-05 22:52:01.455014639 &#43;0800 CST m=&#43;0.001674288

//package entity defines entities used in sdk
package entity 

import "github.com/milvus-io/milvus-sdk-go/internal/proto/schema"


// columnFloatVector generated columns type for FloatVector
type columnFloatVector struct {
	name   string
	dim    int
	values [][]float32
}

func (c *columnFloatVector) Name() string {
	return c.name
}

func (c *columnFloatVector) Type() FieldType {
	return FieldTypeFloatVector
}

func (c *columnFloatVector) FieldData() *schema.FieldData {
	fd := &schema.FieldData{
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

func NewColumnFloatVector(name string,dim int, values [][]float32) Column {
	return &columnFloatVector {
		name:   name,
		dim:    dim,
		values: values,
	}
}
