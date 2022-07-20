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
	"errors"
	"fmt"
	"go/ast"
	"reflect"
	"strconv"
	"strings"
)

const (
	// MilvusTag struct tag const for milvus row based struct
	MilvusTag = `milvus`

	// MilvusTagSep struct tag const for attribute separator
	MilvusTagSep = `;`

	//MilvusTagName struct tag const for field name
	MilvusTagName = `NAME`

	// VectorDimTag struct tag const for vector dimension
	VectorDimTag = `DIM`

	// MilvusPrimaryKey struct tag const for primary key indicator
	MilvusPrimaryKey = `PRIMARY_KEY`

	// MilvusAutoID struct tag const for auto id indicator
	MilvusAutoID = `AUTO_ID`

	// DimMax dimension max value
	DimMax = 65535
)

// Row is the interface for milvus row based data
type Row interface {
	Collection() string
	Partition() string
	Description() string
}

// RowBase row base, returns default collection, partition name which is empty string
type RowBase struct{}

// Collection row base default collection name, which is empty string
// when empty string is passed, the parent struct type name is used
func (b RowBase) Collection() string {
	return ""
}

// Partition row base default partition name, which is empty string
// when empty string is passed, the default partition is used, which currently is named `_default`
func (b RowBase) Partition() string {
	return ""
}

// Description implement Row interface, default value is empty string
func (b RowBase) Description() string {
	return ""
}

// ParseSchema parse Schema from row interface
func ParseSchema(r Row) (*Schema, error) {
	sch := &Schema{
		CollectionName: r.Collection(),
		Description:    r.Description(),
	}
	t := reflect.TypeOf(r)
	if t.Kind() == reflect.Array || t.Kind() == reflect.Slice || t.Kind() == reflect.Ptr {
		t = t.Elem()
	}
	if t.Kind() != reflect.Struct {
		return nil, fmt.Errorf("unsupported data type: %+v", r)
	}

	// Collection method not overwrited, try use Row type name
	if sch.CollectionName == "" {
		sch.CollectionName = t.Name()
		if sch.CollectionName == "" {
			return nil, errors.New("collection name not provided")
		}
	}
	sch.Fields = make([]*Field, 0, t.NumField())
	for i := 0; i < t.NumField(); i++ {
		f := t.Field(i)
		// ignore anonymous field for now
		if f.Anonymous || !ast.IsExported(f.Name) {
			continue
		}

		field := &Field{
			Name: f.Name,
		}
		ft := f.Type
		if f.Type.Kind() == reflect.Ptr {
			ft = ft.Elem()
		}
		fv := reflect.New(ft)
		tagSettings := ParseTagSetting(f.Tag.Get(MilvusTag), MilvusTagSep)
		if _, has := tagSettings[MilvusPrimaryKey]; has {
			field.PrimaryKey = true
		}
		if _, has := tagSettings[MilvusAutoID]; has {
			field.AutoID = true
		}
		if name, has := tagSettings[MilvusTagName]; has {
			field.Name = name
		}
		switch reflect.Indirect(fv).Kind() {
		case reflect.Bool:
			field.DataType = FieldTypeBool
		case reflect.Int8:
			field.DataType = FieldTypeInt8
		case reflect.Int16:
			field.DataType = FieldTypeInt16
		case reflect.Int32:
			field.DataType = FieldTypeInt32
		case reflect.Int64:
			field.DataType = FieldTypeInt64
		case reflect.Float32:
			field.DataType = FieldTypeFloat
		case reflect.Float64:
			field.DataType = FieldTypeDouble
		case reflect.String:
			field.DataType = FieldTypeString
		case reflect.Array:
			arrayLen := ft.Len()
			elemType := ft.Elem()
			switch elemType.Kind() {
			case reflect.Uint8:
				field.DataType = FieldTypeBinaryVector
				//TODO maybe override by tag settings, when dim is not multiplier of 8
				field.TypeParams = map[string]string{
					TypeParamDim: strconv.FormatInt(int64(arrayLen*8), 10),
				}
			case reflect.Float32:
				field.DataType = FieldTypeFloatVector
				field.TypeParams = map[string]string{
					TypeParamDim: strconv.FormatInt(int64(arrayLen), 10),
				}
			default:
				return nil, fmt.Errorf("field %s is array of %v, which is not supported", f.Name, elemType)
			}
		case reflect.Slice:
			dimStr, has := tagSettings[VectorDimTag]
			if !has {
				return nil, fmt.Errorf("field %s is slice but dim not provided", f.Name)
			}
			dim, err := strconv.ParseInt(dimStr, 10, 64)
			if err != nil {
				return nil, fmt.Errorf("dim value %s is not valid", dimStr)
			}
			if dim < 1 || dim > DimMax {
				return nil, fmt.Errorf("dim value %d is out of range", dim)
			}
			field.TypeParams = map[string]string{
				TypeParamDim: dimStr,
			}
			elemType := ft.Elem()
			switch elemType.Kind() {
			case reflect.Uint8: // []byte!
				field.DataType = FieldTypeBinaryVector
			case reflect.Float32:
				field.DataType = FieldTypeFloatVector
			default:
				return nil, fmt.Errorf("field %s is slice of %v, which is not supported", f.Name, elemType)
			}
		default:
			return nil, fmt.Errorf("field %s is %v, which is not supported", field.Name, ft)
		}
		sch.Fields = append(sch.Fields, field)
	}

	return sch, nil
}

// ParseTagSetting parses struct tag into map settings
func ParseTagSetting(str string, sep string) map[string]string {
	settings := map[string]string{}
	names := strings.Split(str, sep)

	for i := 0; i < len(names); i++ {
		j := i
		if len(names[j]) > 0 {
			for {
				if names[j][len(names[j])-1] == '\\' {
					i++
					names[j] = names[j][0:len(names[j])-1] + sep + names[i]
					names[i] = ""
				} else {
					break
				}
			}
		}

		values := strings.Split(names[j], ":")
		k := strings.TrimSpace(strings.ToUpper(values[0]))

		if len(values) >= 2 {
			settings[k] = strings.Join(values[1:], ":")
		} else if k != "" {
			settings[k] = k
		}
	}

	return settings
}

// RowsToColumns rows to columns
func RowsToColumns(rows []Row, schemas ...*Schema) ([]Column, error) {
	rowsLen := len(rows)
	if rowsLen == 0 {
		return []Column{}, errors.New("0 length column")
	}

	var sch *Schema
	var err error
	// if schema not provided, try to parse from row
	if len(schemas) == 0 {
		sch, err = ParseSchema(rows[0])
	} else {
		// use first schema provided
		sch = schemas[0]
	}
	if err != nil {
		return []Column{}, err
	}
	nameColumns := make(map[string]Column)
	for _, field := range sch.Fields {
		switch field.DataType {
		case FieldTypeBool:
			data := make([]bool, 0, rowsLen)
			col := NewColumnBool(field.Name, data)
			nameColumns[field.Name] = col
		case FieldTypeInt8:
			data := make([]int8, 0, rowsLen)
			col := NewColumnInt8(field.Name, data)
			nameColumns[field.Name] = col
		case FieldTypeInt16:
			data := make([]int16, 0, rowsLen)
			col := NewColumnInt16(field.Name, data)
			nameColumns[field.Name] = col
		case FieldTypeInt32:
			data := make([]int32, 0, rowsLen)
			col := NewColumnInt32(field.Name, data)
			nameColumns[field.Name] = col
		case FieldTypeInt64:
			data := make([]int64, 0, rowsLen)
			col := NewColumnInt64(field.Name, data)
			nameColumns[field.Name] = col
		case FieldTypeFloat:
			data := make([]float32, 0, rowsLen)
			col := NewColumnFloat(field.Name, data)
			nameColumns[field.Name] = col
		case FieldTypeDouble:
			data := make([]float64, 0, rowsLen)
			col := NewColumnDouble(field.Name, data)
			nameColumns[field.Name] = col
		case FieldTypeString:
			data := make([]string, 0, rowsLen)
			col := NewColumnString(field.Name, data)
			nameColumns[field.Name] = col
		case FieldTypeFloatVector:
			data := make([][]float32, 0, rowsLen)
			dimStr, has := field.TypeParams[TypeParamDim]
			if !has {
				return []Column{}, errors.New("vector field with no dim")
			}
			dim, err := strconv.ParseInt(dimStr, 10, 64)
			if err != nil {
				return []Column{}, fmt.Errorf("vector field with bad format dim: %s", err.Error())
			}
			col := NewColumnFloatVector(field.Name, int(dim), data)
			nameColumns[field.Name] = col
		case FieldTypeBinaryVector:
			data := make([][]byte, 0, rowsLen)
			dimStr, has := field.TypeParams[TypeParamDim]
			if !has {
				return []Column{}, errors.New("vector field with no dim")
			}
			dim, err := strconv.ParseInt(dimStr, 10, 64)
			if err != nil {
				return []Column{}, fmt.Errorf("vector field with bad format dim: %s", err.Error())
			}
			col := NewColumnBinaryVector(field.Name, int(dim), data)
			nameColumns[field.Name] = col
		}
	}
	for _, row := range rows {
		// collection schema name need not to be same, since receiver could has other names
		v := reflect.ValueOf(row)
		if v.Kind() == reflect.Ptr {
			v = v.Elem()
		}

		for idx, field := range sch.Fields {
			column := nameColumns[field.Name]

			fv := fieldFromNameTag(v, field.Name)
			if !fv.IsValid() {
				return []Column{}, fmt.Errorf("row %d does not has field %s", idx, field.Name)
			}
			if fv.Kind() == reflect.Array { // change to slice
				fv = fv.Slice(0, fv.Len()-1)
			}
			err := column.AppendValue(fv.Interface())
			if err != nil {
				return []Column{}, err
			}
		}

	}
	columns := make([]Column, 0, len(nameColumns))
	for _, column := range nameColumns {
		columns = append(columns, column)
	}
	return columns, nil
}

func fieldFromNameTag(v reflect.Value, name string) reflect.Value {
	// tag has higher priority
	for i := 0; i < v.NumField(); i++ {
		tag := v.Type().Field(i).Tag.Get(MilvusTag)
		if tag == "" {
			continue
		}
		settings := ParseTagSetting(tag, MilvusTagSep)
		fn, has := settings[MilvusTagName]
		if has && fn == name {
			return v.Field(i)
		}
	}
	// try use name directly
	return v.FieldByName(name)
}
