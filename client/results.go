package client

import (
	"go/ast"
	"reflect"

	"github.com/cockroachdb/errors"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
)

// SearchResult contains the result from Search api of client
// IDs is the auto generated id values for the entities
// Fields contains the data of `outputFieleds` specified or all columns if non
// Scores is actually the distance between the vector current record contains and the search target vector
type SearchResult struct {
	// internal schema for unmarshaling
	sch *entity.Schema

	ResultCount  int // the returning entry count
	GroupByValue entity.Column
	IDs          entity.Column // auto generated id, can be mapped to the columns from `Insert` API
	Fields       ResultSet     //[]entity.Column // output field data
	Scores       []float32     // distance to the target vector
	Err          error         // search error if any
}

func (sr *SearchResult) Slice(start, end int) *SearchResult {
	id := end

	result := &SearchResult{
		IDs:    sr.IDs.Slice(start, end),
		Fields: sr.Fields.Slice(start, end),

		Err: sr.Err,
	}
	if sr.GroupByValue != nil {
		result.GroupByValue = sr.GroupByValue.Slice(start, end)
	}

	result.ResultCount = result.IDs.Len()

	l := len(sr.Scores)
	if start > l {
		start = l
	}
	if id > l || id < 0 {
		id = l
	}
	result.Scores = sr.Scores[start:id]

	return result
}

func (sr *SearchResult) Unmarshal(receiver interface{}) (err error) {
	err = sr.Fields.Unmarshal(receiver)
	if err != nil {
		return err
	}
	return sr.fillPKEntry(receiver)
}

func (sr *SearchResult) fillPKEntry(receiver interface{}) (err error) {
	defer func() {
		if x := recover(); x != nil {
			err = errors.Newf("failed to unmarshal result set: %v", x)
		}
	}()
	rr := reflect.ValueOf(receiver)

	if rr.Kind() == reflect.Ptr {
		if rr.IsNil() && rr.CanAddr() {
			rr.Set(reflect.New(rr.Type().Elem()))
		}
		rr = rr.Elem()
	}

	rt := rr.Type()
	rv := rr

	switch rt.Kind() {
	case reflect.Slice:
		pkField := sr.sch.PKField()

		et := rt.Elem()
		for et.Kind() == reflect.Ptr {
			et = et.Elem()
		}

		candidates := parseCandidates(et)
		candi, ok := candidates[pkField.Name]
		if !ok {
			// pk field not found in struct, skip
			return nil
		}
		for i := 0; i < sr.IDs.Len(); i++ {
			row := rv.Index(i)
			for row.Kind() == reflect.Ptr {
				row = row.Elem()
			}

			val, err := sr.IDs.Get(i)
			if err != nil {
				return err
			}
			row.Field(candi).Set(reflect.ValueOf(val))
		}
		rr.Set(rv)
	default:
		return errors.Newf("receiver need to be slice or array but get %v", rt.Kind())
	}
	return nil
}

// ResultSet is an alias type for column slice.
type ResultSet []entity.Column

func (rs ResultSet) Len() int {
	if len(rs) == 0 {
		return 0
	}
	return rs[0].Len()
}

func (rs ResultSet) Slice(start, end int) ResultSet {
	result := make([]entity.Column, 0, len(rs))
	for _, col := range rs {
		result = append(result, col.Slice(start, end))
	}
	return result
}

// GetColumn returns column with provided field name.
func (rs ResultSet) GetColumn(fieldName string) entity.Column {
	for _, column := range rs {
		if column.Name() == fieldName {
			return column
		}
	}
	return nil
}

func (rs ResultSet) Unmarshal(receiver interface{}) (err error) {
	defer func() {
		if x := recover(); x != nil {
			err = errors.Newf("failed to unmarshal result set: %v", x)
		}
	}()
	rr := reflect.ValueOf(receiver)

	if rr.Kind() == reflect.Ptr {
		if rr.IsNil() && rr.CanAddr() {
			rr.Set(reflect.New(rr.Type().Elem()))
		}
		rr = rr.Elem()
	}

	rt := rr.Type()
	rv := rr

	switch rt.Kind() {
	// TODO maybe support Array and just fill data
	// case reflect.Array:
	case reflect.Slice:
		et := rt.Elem()
		if et.Kind() != reflect.Ptr {
			return errors.Newf("receiver must be slice of pointers but get: %v", et.Kind())
		}
		for et.Kind() == reflect.Ptr {
			et = et.Elem()
		}
		for i := 0; i < rs.Len(); i++ {
			data := reflect.New(et)
			err := rs.fillData(data.Elem(), et, i)
			if err != nil {
				return err
			}
			rv = reflect.Append(rv, data)
		}
		rr.Set(rv)
	default:
		return errors.Newf("receiver need to be slice or array but get %v", rt.Kind())
	}
	return nil
}

func parseCandidates(dataType reflect.Type) map[string]int {
	result := make(map[string]int)
	for i := 0; i < dataType.NumField(); i++ {
		f := dataType.Field(i)
		// ignore anonymous field for now
		if f.Anonymous || !ast.IsExported(f.Name) {
			continue
		}

		name := f.Name
		tag := f.Tag.Get(entity.MilvusTag)
		tagSettings := entity.ParseTagSetting(tag, entity.MilvusTagSep)
		if tagName, has := tagSettings[entity.MilvusTagName]; has {
			name = tagName
		}

		result[name] = i
	}
	return result
}

func (rs ResultSet) fillData(data reflect.Value, dataType reflect.Type, idx int) error {
	m := parseCandidates(dataType)
	for i := 0; i < len(rs); i++ {
		name := rs[i].Name()
		fidx, ok := m[name]
		if !ok {
			// maybe return error
			continue
		}
		val, err := rs[i].Get(idx)
		if err != nil {
			return err
		}
		// TODO check datatype
		data.Field(fidx).Set(reflect.ValueOf(val))
	}
	return nil
}
