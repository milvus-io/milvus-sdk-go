package common

import (
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
)

// -- create field --

// CreateFieldOption is an option that is used to modify entity.Schema
type CreateFieldOption func(field *entity.Field)

func WithIsPrimaryKey(isPrimaryKey bool) CreateFieldOption {
	return func(field *entity.Field) {
		field.WithIsPrimaryKey(isPrimaryKey)
	}
}

func WithAutoID(autoID bool) CreateFieldOption {
	return func(field *entity.Field) {
		field.WithIsAutoID(autoID)
	}
}

func WithFieldDescription(desc string) CreateFieldOption {
	return func(field *entity.Field) {
		field.WithDescription(desc)
	}
}

func WithIsPartitionKey(isPartitionKey bool) CreateFieldOption {
	return func(field *entity.Field) {
		field.WithIsPartitionKey(isPartitionKey)
	}
}

func WithDim(dim int64) CreateFieldOption {
	return func(field *entity.Field) {
		field.WithDim(dim)
	}
}

func WithMaxLength(maxLen int64) CreateFieldOption {
	return func(field *entity.Field) {
		field.WithMaxLength(maxLen)
	}
}

func WithTypeParams(key string, value string) CreateFieldOption {
	return func(field *entity.Field) {
		field.WithTypeParams(key, value)
	}
}

func GenField(name string, fieldType entity.FieldType, opts ...CreateFieldOption) *entity.Field {
	if name == "" {
		name = fieldType.Name() + GenRandomString(2)
	}
	field := entity.NewField().
		WithName(name).
		WithDataType(fieldType)

	for _, opt := range opts {
		opt(field)
	}
	return field
}

// -- create field --

// -- create schema --

// CreateSchemaOption is an option that is used to modify entity.Schema
type CreateSchemaOption func(schema *entity.Schema)

func WithDescription(desc string) CreateSchemaOption {
	return func(schema *entity.Schema) {
		schema.Description = desc
	}
}

func WithEnableDynamicField(enableDF bool) CreateSchemaOption {
	return func(schema *entity.Schema) {
		schema.EnableDynamicField = enableDF
	}
}

// gen schema
func GenSchema(name string, autoID bool, fields []*entity.Field, opts ...CreateSchemaOption) *entity.Schema {
	schema := &entity.Schema{
		CollectionName: name,
		AutoID:         autoID,
		Fields:         fields,
	}
	for _, opt := range opts {
		opt(schema)
	}
	return schema
}

// -- create schema --
