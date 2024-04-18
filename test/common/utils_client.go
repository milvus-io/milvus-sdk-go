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

func WithElementType(eleType entity.FieldType) CreateFieldOption {
	return func(field *entity.Field) {
		field.WithElementType(eleType)
	}
}

func WithMaxCapacity(maxCap int64) CreateFieldOption {
	return func(field *entity.Field) {
		field.WithMaxCapacity(maxCap)
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

// GenSchema gen schema
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

// GenColumnDataOption -- create column data --
type GenColumnDataOption func(opt *genDataOpt)
type genDataOpt struct {
	dim          int64
	ElementType  entity.FieldType
	capacity     int64
	maxLenSparse int
}

func WithVectorDim(dim int64) GenColumnDataOption {
	return func(opt *genDataOpt) {
		opt.dim = dim
	}
}

func WithArrayElementType(eleType entity.FieldType) GenColumnDataOption {
	return func(opt *genDataOpt) {
		opt.ElementType = eleType
	}
}

func WithArrayCapacity(capacity int64) GenColumnDataOption {
	return func(opt *genDataOpt) {
		opt.capacity = capacity
	}
}

func WithSparseVectorLen(length int) GenColumnDataOption {
	return func(opt *genDataOpt) {
		opt.maxLenSparse = length
	}
}

// -- create column data --
