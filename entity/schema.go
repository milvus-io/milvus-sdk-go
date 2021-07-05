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

// pakcage entity defines entities used in sdk
package entity

import (
	"github.com/milvus-io/milvus-sdk-go/internal/proto/common"
	"github.com/milvus-io/milvus-sdk-go/internal/proto/schema"
)

// Schema represents schema info of collection in milvus
type Schema struct {
	CollectionName string
	Description    string
	AutoID         bool
	Fields         []*Field
}

// ProtoMessage returns corresponding server.CollectionSchema
func (s *Schema) ProtoMessage() *schema.CollectionSchema {
	r := &schema.CollectionSchema{
		Name:        s.CollectionName,
		Description: s.Description,
		AutoID:      s.AutoID,
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
		s.Fields = append(s.Fields, (&Field{}).ReadProto(fp))
	}
	return s
}

// Field represent field schema in milvus
type Field struct {
	ID          int64  // field id, generated when collection is created, input value is ignored
	Name        string // field name
	PrimaryKey  bool   // is primary key
	Description string
	DataType    FieldType
	TypeParams  map[string]string
}

// ProtoMessage generetes corresponding FieldSchema
func (f *Field) ProtoMessage() *schema.FieldSchema {
	return &schema.FieldSchema{
		FieldID:      f.ID,
		Name:         f.Name,
		Description:  f.Description,
		IsPrimaryKey: f.PrimaryKey,
		DataType:     schema.DataType(f.DataType),
		TypeParams:   mapKvPairs(f.TypeParams),
	}
}

// ReadProto parses FieldSchema
func (f *Field) ReadProto(p *schema.FieldSchema) *Field {
	f.ID = p.GetFieldID()
	f.Name = p.GetName()
	f.PrimaryKey = p.GetIsPrimaryKey()
	f.Description = p.GetDescription()
	f.DataType = FieldType(p.GetDataType())
	f.TypeParams = KvPairsMap(p.GetTypeParams())

	return f
}

func mapKvPairs(m map[string]string) []*common.KeyValuePair {
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
type FieldType int32

// Match schema definition
const (
	//FieldTypeNone zero value place holder
	FieldTypeNone FieldType = 0 // zero value place holder
	//FieldTypeBool field type boolean
	FieldTypeBool FieldType = 1
	//FieldTypeInt8 field type int8
	FieldTypeInt8 FieldType = 2
	//FieldTypeInt16 field type int16
	FieldTypeInt16 FieldType = 3
	//FieldTypeInt32 field type int32
	FieldTypeInt32 FieldType = 4
	//FIeldTypeInt64 field type int64
	FieldTypeInt64 FieldType = 5
	//FieldTypeFloat field type float
	FieldTypeFloat FieldType = 10
	//FieldTypeDouble field type double
	FieldTypeDouble FieldType = 11
	//FieldTypeString field type string
	FieldTypeString FieldType = 20
	//FieldTypeBinaryVector field type binary vector
	FieldTypeBinaryVector FieldType = 100
	//FieldTypeFloatVector field type float vector
	FieldTypeFloatVector FieldType = 101
)
