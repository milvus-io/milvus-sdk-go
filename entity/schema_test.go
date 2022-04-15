package entity

import (
	"testing"

	"github.com/milvus-io/milvus-sdk-go/v2/internal/proto/common"
	"github.com/stretchr/testify/assert"
)

func TestCL_CommonCL(t *testing.T) {
	cls := []ConsistencyLevel{
		CL_STRONG,
		CL_BOUNDED,
		CL_SESSION,
		CL_EVENTUALLY,
	}
	for _, cl := range cls {
		assert.EqualValues(t, common.ConsistencyLevel(cl), cl.CommonConsisencyLevel())
	}
}

func TestFieldSchema(t *testing.T) {
	fields := []*Field{
		{
			Name:        "int_field",
			DataType:    FieldTypeInt64,
			AutoID:      true,
			PrimaryKey:  true,
			Description: "int_field desc",
			TypeParams:  map[string]string{},
		},
		{
			Name:        "string_field",
			DataType:    FieldTypeString,
			AutoID:      false,
			PrimaryKey:  true,
			Description: "false_field desc",
			TypeParams:  map[string]string{"max_len": "32"}, // not applied, just testing value
		},
	}

	for _, field := range fields {
		fieldSchema := field.ProtoMessage()
		assert.Equal(t, field.ID, fieldSchema.GetFieldID())
		assert.Equal(t, field.Name, fieldSchema.GetName())
		assert.EqualValues(t, field.DataType, fieldSchema.GetDataType())
		assert.Equal(t, field.AutoID, fieldSchema.GetAutoID())
		assert.Equal(t, field.PrimaryKey, fieldSchema.GetIsPrimaryKey())
		assert.Equal(t, field.Description, fieldSchema.GetDescription())
		assert.Equal(t, field.TypeParams, KvPairsMap(fieldSchema.GetTypeParams()))
		// marshal & unmarshal, still equals
		nf := &Field{}
		nf = nf.ReadProto(fieldSchema)
		assert.Equal(t, field.ID, nf.ID)
		assert.Equal(t, field.Name, nf.Name)
		assert.EqualValues(t, field.DataType, nf.DataType)
		assert.Equal(t, field.AutoID, nf.AutoID)
		assert.Equal(t, field.PrimaryKey, nf.PrimaryKey)
		assert.Equal(t, field.Description, nf.Description)
		assert.Equal(t, field.TypeParams, nf.TypeParams)
	}
}

func TestSchema(t *testing.T) {
	schemas := []*Schema{
		{
			CollectionName: "test_collection_1",
			AutoID:         false,
			Description:    "test_collection_1 decription",
		},
	}
	for _, sch := range schemas {
		p := sch.ProtoMessage()
		assert.Equal(t, sch.CollectionName, p.GetName())
		assert.Equal(t, sch.AutoID, p.GetAutoID())
		assert.Equal(t, sch.Description, p.GetDescription())

		nsch := &Schema{}
		nsch = nsch.ReadProto(p)

		assert.Equal(t, sch.CollectionName, nsch.CollectionName)
		assert.Equal(t, sch.AutoID, nsch.AutoID)
		assert.Equal(t, sch.Description, nsch.Description)
	}
}
