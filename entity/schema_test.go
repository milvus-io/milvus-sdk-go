package entity

import (
	"testing"

	common "github.com/milvus-io/milvus-proto/go-api/commonpb"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/suite"
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
		NewField().WithName("partition_key").WithDataType(FieldTypeInt32).WithIsPartitionKey(true),
	}

	for _, field := range fields {
		fieldSchema := field.ProtoMessage()
		assert.Equal(t, field.ID, fieldSchema.GetFieldID())
		assert.Equal(t, field.Name, fieldSchema.GetName())
		assert.EqualValues(t, field.DataType, fieldSchema.GetDataType())
		assert.Equal(t, field.AutoID, fieldSchema.GetAutoID())
		assert.Equal(t, field.PrimaryKey, fieldSchema.GetIsPrimaryKey())
		assert.Equal(t, field.IsPartitionKey, fieldSchema.GetIsPartitionKey())
		assert.Equal(t, field.IsDynamic, fieldSchema.GetIsDynamic())
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
		assert.Equal(t, field.IsDynamic, nf.IsDynamic)
		assert.Equal(t, field.IsPartitionKey, nf.IsPartitionKey)
		assert.EqualValues(t, field.TypeParams, nf.TypeParams)
	}

	assert.NotPanics(t, func() {
		(&Field{}).WithTypeParams("a", "b")
	})
}

type SchemaSuite struct {
	suite.Suite
}

func (s *SchemaSuite) TestBasic() {
	cases := []struct {
		tag    string
		input  *Schema
		pkName string
	}{
		{
			"test_collection",
			NewSchema().WithName("test_collection_1").WithDescription("test_collection_1 desc").WithAutoID(false).
				WithField(NewField().WithName("ID").WithDataType(FieldTypeInt64).WithIsPrimaryKey(true)).
				WithField(NewField().WithName("vector").WithDataType(FieldTypeFloatVector).WithDim(128)),
			"ID",
		},
		{
			"dynamic_schema",
			NewSchema().WithName("dynamic_schema").WithDescription("dynamic_schema desc").WithAutoID(true).WithDynamicFieldEnabled(true).
				WithField(NewField().WithName("ID").WithDataType(FieldTypeVarChar).WithMaxLength(256)).
				WithField(NewField().WithName("$meta").WithIsDynamic(true)),
			"",
		},
	}

	for _, c := range cases {
		s.Run(c.tag, func() {
			sch := c.input
			p := sch.ProtoMessage()
			s.Equal(sch.CollectionName, p.GetName())
			s.Equal(sch.AutoID, p.GetAutoID())
			s.Equal(sch.Description, p.GetDescription())
			s.Equal(sch.EnableDynamicField, p.GetEnableDynamicField())
			s.Equal(len(sch.Fields), len(p.GetFields()))

			nsch := &Schema{}
			nsch = nsch.ReadProto(p)

			s.Equal(sch.CollectionName, nsch.CollectionName)
			s.Equal(sch.AutoID, nsch.AutoID)
			s.Equal(sch.Description, nsch.Description)
			s.Equal(sch.EnableDynamicField, nsch.EnableDynamicField)
			s.Equal(len(sch.Fields), len(nsch.Fields))
			s.Equal(c.pkName, sch.PKFieldName())
			s.Equal(c.pkName, nsch.PKFieldName())
		})
	}
}

func TestSchema(t *testing.T) {
	suite.Run(t, new(SchemaSuite))
}
