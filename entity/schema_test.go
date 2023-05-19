package entity

import (
	"testing"

	common "github.com/milvus-io/milvus-proto/go-api/commonpb"
	"github.com/stretchr/testify/assert"
)

func TestCL_CommonCL(t *testing.T) {
	cls := []ConsistencyLevel{
		ClStrong,
		ClBounded,
		ClSession,
		ClEventually,
	}
	for _, cl := range cls {
		assert.EqualValues(t, common.ConsistencyLevel(cl), cl.CommonConsistencyLevel())
	}
}

func TestFieldSchema(t *testing.T) {
	fields := []*Field{
		NewField().WithName("int_field").WithDataType(FieldTypeInt64).WithIsAutoID(true).WithIsPrimaryKey(true).WithDescription("int_field desc"),
		NewField().WithName("string_field").WithDataType(FieldTypeString).WithIsAutoID(false).WithIsPrimaryKey(true).WithIsDynamic(false).WithTypeParams("max_len", "32").WithDescription("string_field desc"),
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
		assert.EqualValues(t, field.TypeParams, nf.TypeParams)
	}

	assert.NotPanics(t, func() {
		(&Field{}).WithTypeParams("a", "b")
	})
}

func TestSchema(t *testing.T) {
	schemas := []*Schema{
		NewSchema().WithName("test_collection_1").WithDescription("test_collection_1 desc").WithAutoID(false),
		NewSchema().WithName("dynamic_schema").WithDescription("dynamic_schema desc").WithAutoID(true).WithDynamicFieldEnabled(true),
	}
	for _, sch := range schemas {
		p := sch.ProtoMessage()
		assert.Equal(t, sch.CollectionName, p.GetName())
		assert.Equal(t, sch.AutoID, p.GetAutoID())
		assert.Equal(t, sch.Description, p.GetDescription())
		assert.Equal(t, sch.EnableDynamicField, p.GetEnableDynamicField())

		nsch := &Schema{}
		nsch = nsch.ReadProto(p)

		assert.Equal(t, sch.CollectionName, nsch.CollectionName)
		assert.Equal(t, sch.AutoID, nsch.AutoID)
		assert.Equal(t, sch.Description, nsch.Description)
		assert.Equal(t, sch.EnableDynamicField, nsch.EnableDynamicField)
	}
}
