// Copyright (C) 2019-2021 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License.

package entity

import (
	"strconv"

	common "github.com/milvus-io/milvus-proto/go-api/v2/commonpb"
	schema "github.com/milvus-io/milvus-proto/go-api/v2/schemapb"
)

const (
	// TypeParamDim is the const for field type param dimension
	TypeParamDim = "dim"

	// TypeParamMaxLength is the const for varchar type maximal length
	TypeParamMaxLength = "max_length"

	// TypeParamMaxCapacity is the const for array type max capacity
	TypeParamMaxCapacity = `max_capacity`

	// ClStrong strong consistency level
	ClStrong ConsistencyLevel = ConsistencyLevel(common.ConsistencyLevel_Strong)
	// ClBounded bounded consistency level with default tolerance of 5 seconds
	ClBounded ConsistencyLevel = ConsistencyLevel(common.ConsistencyLevel_Bounded)
	// ClSession session consistency level
	ClSession ConsistencyLevel = ConsistencyLevel(common.ConsistencyLevel_Session)
	// ClEvenually eventually consistency level
	ClEventually ConsistencyLevel = ConsistencyLevel(common.ConsistencyLevel_Eventually)
	// ClCustomized customized consistency level and users pass their own `guarantee_timestamp`.
	ClCustomized ConsistencyLevel = ConsistencyLevel(common.ConsistencyLevel_Customized)
)

// ConsistencyLevel enum type for collection Consistency Level
type ConsistencyLevel common.ConsistencyLevel

// CommonConsistencyLevel returns corresponding common.ConsistencyLevel
func (cl ConsistencyLevel) CommonConsistencyLevel() common.ConsistencyLevel {
	return common.ConsistencyLevel(cl)
}

// Schema represents schema info of collection in milvus
type Schema struct {
	CollectionName     string
	Description        string
	AutoID             bool
	Fields             []*Field
	EnableDynamicField bool
	pkField            *Field
}

// NewSchema creates an empty schema object.
func NewSchema() *Schema {
	return &Schema{}
}

// WithName sets the name value of schema, returns schema itself.
func (s *Schema) WithName(name string) *Schema {
	s.CollectionName = name
	return s
}

// WithDescription sets the description value of schema, returns schema itself.
func (s *Schema) WithDescription(desc string) *Schema {
	s.Description = desc
	return s
}

func (s *Schema) WithAutoID(autoID bool) *Schema {
	s.AutoID = autoID
	return s
}

func (s *Schema) WithDynamicFieldEnabled(dynamicEnabled bool) *Schema {
	s.EnableDynamicField = dynamicEnabled
	return s
}

// WithField adds a field into schema and returns schema itself.
func (s *Schema) WithField(f *Field) *Schema {
	if f.PrimaryKey {
		s.pkField = f
	}
	s.Fields = append(s.Fields, f)
	return s
}

// ProtoMessage returns corresponding server.CollectionSchema
func (s *Schema) ProtoMessage() *schema.CollectionSchema {
	r := &schema.CollectionSchema{
		Name:               s.CollectionName,
		Description:        s.Description,
		AutoID:             s.AutoID,
		EnableDynamicField: s.EnableDynamicField,
	}
	r.Fields = make([]*schema.FieldSchema, 0, len(s.Fields))
	for _, field := range s.Fields {
		r.Fields = append(r.Fields, field.ProtoMessage())
	}
	return r
}

// ReadProto parses proto Collection Schema
func (s *Schema) ReadProto(p *schema.CollectionSchema) *Schema {
	s.AutoID = p.GetAutoID()
	s.Description = p.GetDescription()
	s.CollectionName = p.GetName()
	s.Fields = make([]*Field, 0, len(p.GetFields()))
	for _, fp := range p.GetFields() {
		field := NewField().ReadProto(fp)
		if field.PrimaryKey {
			s.pkField = field
		}
		s.Fields = append(s.Fields, field)
	}
	s.EnableDynamicField = p.GetEnableDynamicField()
	return s
}

// PKFieldName returns pk field name for this schema.
func (s *Schema) PKFieldName() string {
	if s.pkField == nil {
		return ""
	}
	return s.pkField.Name
}

// PKField returns PK Field schema for this schema.
func (s *Schema) PKField() *Field {
	return s.pkField
}

// Field represent field schema in milvus
type Field struct {
	ID             int64  // field id, generated when collection is created, input value is ignored
	Name           string // field name
	PrimaryKey     bool   // is primary key
	AutoID         bool   // is auto id
	Description    string
	DataType       FieldType
	TypeParams     map[string]string
	IndexParams    map[string]string
	IsDynamic      bool
	IsPartitionKey bool
	ElementType    FieldType
}

// ProtoMessage generates corresponding FieldSchema
func (f *Field) ProtoMessage() *schema.FieldSchema {
	return &schema.FieldSchema{
		FieldID:        f.ID,
		Name:           f.Name,
		Description:    f.Description,
		IsPrimaryKey:   f.PrimaryKey,
		AutoID:         f.AutoID,
		DataType:       schema.DataType(f.DataType),
		TypeParams:     MapKvPairs(f.TypeParams),
		IndexParams:    MapKvPairs(f.IndexParams),
		IsDynamic:      f.IsDynamic,
		IsPartitionKey: f.IsPartitionKey,
		ElementType:    schema.DataType(f.ElementType),
	}
}

// NewField creates a new Field with map initialized.
func NewField() *Field {
	return &Field{
		TypeParams:  make(map[string]string),
		IndexParams: make(map[string]string),
	}
}

func (f *Field) WithName(name string) *Field {
	f.Name = name
	return f
}

func (f *Field) WithDescription(desc string) *Field {
	f.Description = desc
	return f
}

func (f *Field) WithDataType(dataType FieldType) *Field {
	f.DataType = dataType
	return f
}

func (f *Field) WithIsPrimaryKey(isPrimaryKey bool) *Field {
	f.PrimaryKey = isPrimaryKey
	return f
}

func (f *Field) WithIsAutoID(isAutoID bool) *Field {
	f.AutoID = isAutoID
	return f
}

func (f *Field) WithIsDynamic(isDynamic bool) *Field {
	f.IsDynamic = isDynamic
	return f
}

func (f *Field) WithIsPartitionKey(isPartitionKey bool) *Field {
	f.IsPartitionKey = isPartitionKey
	return f
}

/*
func (f *Field) WithDefaultValueBool(defaultValue bool) *Field {
	f.DefaultValue = &schema.ValueField{
		Data: &schema.ValueField_BoolData{
			BoolData: defaultValue,
		},
	}
	return f
}

func (f *Field) WithDefaultValueInt(defaultValue int32) *Field {
	f.DefaultValue = &schema.ValueField{
		Data: &schema.ValueField_IntData{
			IntData: defaultValue,
		},
	}
	return f
}

func (f *Field) WithDefaultValueLong(defaultValue int64) *Field {
	f.DefaultValue = &schema.ValueField{
		Data: &schema.ValueField_LongData{
			LongData: defaultValue,
		},
	}
	return f
}

func (f *Field) WithDefaultValueFloat(defaultValue float32) *Field {
	f.DefaultValue = &schema.ValueField{
		Data: &schema.ValueField_FloatData{
			FloatData: defaultValue,
		},
	}
	return f
}

func (f *Field) WithDefaultValueDouble(defaultValue float64) *Field {
	f.DefaultValue = &schema.ValueField{
		Data: &schema.ValueField_DoubleData{
			DoubleData: defaultValue,
		},
	}
	return f
}

func (f *Field) WithDefaultValueString(defaultValue string) *Field {
	f.DefaultValue = &schema.ValueField{
		Data: &schema.ValueField_StringData{
			StringData: defaultValue,
		},
	}
	return f
}*/

func (f *Field) WithTypeParams(key string, value string) *Field {
	if f.TypeParams == nil {
		f.TypeParams = make(map[string]string)
	}
	f.TypeParams[key] = value
	return f
}

func (f *Field) WithDim(dim int64) *Field {
	if f.TypeParams == nil {
		f.TypeParams = make(map[string]string)
	}
	f.TypeParams[TypeParamDim] = strconv.FormatInt(dim, 10)
	return f
}

func (f *Field) WithMaxLength(maxLen int64) *Field {
	if f.TypeParams == nil {
		f.TypeParams = make(map[string]string)
	}
	f.TypeParams[TypeParamMaxLength] = strconv.FormatInt(maxLen, 10)
	return f
}

func (f *Field) WithElementType(eleType FieldType) *Field {
	f.ElementType = eleType
	return f
}

func (f *Field) WithMaxCapacity(maxCap int64) *Field {
	if f.TypeParams == nil {
		f.TypeParams = make(map[string]string)
	}
	f.TypeParams[TypeParamMaxCapacity] = strconv.FormatInt(maxCap, 10)
	return f
}

// ReadProto parses FieldSchema
func (f *Field) ReadProto(p *schema.FieldSchema) *Field {
	f.ID = p.GetFieldID()
	f.Name = p.GetName()
	f.PrimaryKey = p.GetIsPrimaryKey()
	f.AutoID = p.GetAutoID()
	f.Description = p.GetDescription()
	f.DataType = FieldType(p.GetDataType())
	f.TypeParams = KvPairsMap(p.GetTypeParams())
	f.IndexParams = KvPairsMap(p.GetIndexParams())
	f.IsDynamic = p.GetIsDynamic()
	f.IsPartitionKey = p.GetIsPartitionKey()
	f.ElementType = FieldType(p.GetElementType())

	return f
}

// MapKvPairs converts map into common.KeyValuePair slice
func MapKvPairs(m map[string]string) []*common.KeyValuePair {
	pairs := make([]*common.KeyValuePair, 0, len(m))
	for k, v := range m {
		pairs = append(pairs, &common.KeyValuePair{
			Key:   k,
			Value: v,
		})
	}
	return pairs
}

// KvPairsMap converts common.KeyValuePair slices into map
func KvPairsMap(kvps []*common.KeyValuePair) map[string]string {
	m := make(map[string]string)
	for _, kvp := range kvps {
		m[kvp.Key] = kvp.Value
	}
	return m
}

// FieldType field data type alias type
// used in go:generate trick, DO NOT modify names & string
type FieldType int32

// Name returns field type name
func (t FieldType) Name() string {
	switch t {
	case FieldTypeBool:
		return "Bool"
	case FieldTypeInt8:
		return "Int8"
	case FieldTypeInt16:
		return "Int16"
	case FieldTypeInt32:
		return "Int32"
	case FieldTypeInt64:
		return "Int64"
	case FieldTypeFloat:
		return "Float"
	case FieldTypeDouble:
		return "Double"
	case FieldTypeString:
		return "String"
	case FieldTypeVarChar:
		return "VarChar"
	case FieldTypeArray:
		return "Array"
	case FieldTypeJSON:
		return "JSON"
	case FieldTypeBinaryVector:
		return "BinaryVector"
	case FieldTypeFloatVector:
		return "FloatVector"
	case FieldTypeFloat16Vector:
		return "Float16Vector"
	case FieldTypeBFloat16Vector:
		return "BFloat16Vector"
	default:
		return "undefined"
	}
}

// String returns field type
func (t FieldType) String() string {
	switch t {
	case FieldTypeBool:
		return "bool"
	case FieldTypeInt8:
		return "int8"
	case FieldTypeInt16:
		return "int16"
	case FieldTypeInt32:
		return "int32"
	case FieldTypeInt64:
		return "int64"
	case FieldTypeFloat:
		return "float32"
	case FieldTypeDouble:
		return "float64"
	case FieldTypeString:
		return "string"
	case FieldTypeVarChar:
		return "string"
	case FieldTypeArray:
		return "Array"
	case FieldTypeJSON:
		return "JSON"
	case FieldTypeBinaryVector:
		return "[]byte"
	case FieldTypeFloatVector:
		return "[]float32"
	case FieldTypeFloat16Vector:
		return "[]byte"
	case FieldTypeBFloat16Vector:
		return "[]byte"
	default:
		return "undefined"
	}
}

// PbFieldType represents FieldType corresponding schema pb type
func (t FieldType) PbFieldType() (string, string) {
	switch t {
	case FieldTypeBool:
		return "Bool", "bool"
	case FieldTypeInt8:
		fallthrough
	case FieldTypeInt16:
		fallthrough
	case FieldTypeInt32:
		return "Int", "int32"
	case FieldTypeInt64:
		return "Long", "int64"
	case FieldTypeFloat:
		return "Float", "float32"
	case FieldTypeDouble:
		return "Double", "float64"
	case FieldTypeString:
		return "String", "string"
	case FieldTypeVarChar:
		return "VarChar", "string"
	case FieldTypeJSON:
		return "JSON", "JSON"
	case FieldTypeBinaryVector:
		return "[]byte", ""
	case FieldTypeFloatVector:
		return "[]float32", ""
	case FieldTypeFloat16Vector:
		return "[]byte", ""
	case FieldTypeBFloat16Vector:
		return "[]byte", ""
	default:
		return "undefined", ""
	}
}

// Match schema definition
const (
	// FieldTypeNone zero value place holder
	FieldTypeNone FieldType = 0 // zero value place holder
	// FieldTypeBool field type boolean
	FieldTypeBool FieldType = 1
	// FieldTypeInt8 field type int8
	FieldTypeInt8 FieldType = 2
	// FieldTypeInt16 field type int16
	FieldTypeInt16 FieldType = 3
	// FieldTypeInt32 field type int32
	FieldTypeInt32 FieldType = 4
	// FieldTypeInt64 field type int64
	FieldTypeInt64 FieldType = 5
	// FieldTypeFloat field type float
	FieldTypeFloat FieldType = 10
	// FieldTypeDouble field type double
	FieldTypeDouble FieldType = 11
	// FieldTypeString field type string
	FieldTypeString FieldType = 20
	// FieldTypeVarChar field type varchar
	FieldTypeVarChar FieldType = 21 // variable-length strings with a specified maximum length
	// FieldTypeArray field type Array
	FieldTypeArray FieldType = 22
	// FieldTypeJSON field type JSON
	FieldTypeJSON FieldType = 23
	// FieldTypeBinaryVector field type binary vector
	FieldTypeBinaryVector FieldType = 100
	// FieldTypeFloatVector field type float vector
	FieldTypeFloatVector FieldType = 101
	// FieldTypeBinaryVector field type float16 vector
	FieldTypeFloat16Vector FieldType = 102
	// FieldTypeBinaryVector field type bf16 vector
	FieldTypeBFloat16Vector FieldType = 103
	// FieldTypeBinaryVector field type sparse vector
	FieldTypeSparseVector FieldType = 104
)
