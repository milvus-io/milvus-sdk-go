package client

import (
	"errors"
	"fmt"
	"reflect"

	"github.com/milvus-io/milvus-sdk-go/v2/entity"
	"github.com/milvus-io/milvus-sdk-go/v2/internal/proto/schema"
)

type SearchResultByRows struct {
	ResultCount int
	Scores      []float32
	Rows        []entity.Row
	Err         error
}

func SearchResultToRows(sch *entity.Schema, results *schema.SearchResultData, t reflect.Type) {
	offset := 0
	sr := make([]SearchResultByRows, 0, results.GetNumQueries())
	fieldDataList := results.GetFieldsData()
	nameFieldData := make(map[string]*schema.FieldData)
	for _, fieldData := range fieldDataList {
		nameFieldData[fieldData.FieldName] = fieldData
	}
	ids := results.GetIds()
	for i := 0; i < int(results.GetNumQueries()); i++ {
		rc := int(results.GetTopks()[i]) // result entry count for current query
		entry := SearchResultByRows{
			ResultCount: rc,
			Rows:        make([]entity.Row, 0, rc),
			Scores:      results.GetScores()[offset:rc],
		}
		for j := 0; j < rc; j++ {
			p := reflect.New(t)
			v := p.Elem()

			//extract primary field logic
			for _, field := range sch.Fields {
				f := v.FieldByName(field.Name)
				if f.IsNil() {
					//TODO check field in output list
					continue
				}
				if field.PrimaryKey {
					switch f.Kind() {
					case reflect.Int64:
						intIds := ids.GetIntId()
						if intIds == nil {
							entry.Err = fmt.Errorf("field %s is int64, but id column is not", field.Name)
							break
						}
						f.SetInt(intIds.GetData()[offset+j])
					case reflect.String:
						strIds := ids.GetStrId()
						if strIds == nil {
							entry.Err = fmt.Errorf("field %s is string ,but id column is not", field.Name)
							break
						}
						f.SetString(strIds.GetData()[offset+j])
					default:
						entry.Err = fmt.Errorf("field %s is not valid primary key", field.Name)
						break
					}
					continue
				}

				//fieldDataList
				fieldData, has := nameFieldData[field.Name]
				if !has || fieldData == nil {
					//TODO if check field in output list
					continue
				}

				offset += rc
				sr = append(sr, entry)
			}

		}
	}
}

var (
	ErrFieldTypeNotMatch = errors.New("field type not matched")
)

func SetFieldValue(field *entity.Field, f reflect.Value, fieldData *schema.FieldData, idx int) error {
	scalars := fieldData.GetScalars()
	vectors := fieldData.GetVectors()
	switch field.DataType {
	case entity.FieldTypeBool:
		if f.Kind() != reflect.Bool {
			return ErrFieldTypeNotMatch
		}
		if scalars == nil {
			return ErrFieldTypeNotMatch
		}
		data := scalars.GetBoolData()
		if data == nil {
			return ErrFieldTypeNotMatch
		}
		f.SetBool(data.Data[idx])
	case entity.FieldTypeInt8:
		if f.Kind() != reflect.Int8 {
			return ErrFieldTypeNotMatch
		}
		if scalars == nil {
			return ErrFieldTypeNotMatch
		}
		data := scalars.GetIntData()
		if data == nil {
			return ErrFieldTypeNotMatch
		}
		f.SetInt(int64(data.Data[idx]))
	case entity.FieldTypeInt16:
		if f.Kind() != reflect.Int16 {
			return ErrFieldTypeNotMatch
		}
		if scalars == nil {
			return ErrFieldTypeNotMatch
		}
		data := scalars.GetIntData()
		f.SetInt(int64(data.Data[idx]))
	case entity.FieldTypeInt32:
		if f.Kind() != reflect.Int32 {
			return ErrFieldTypeNotMatch
		}
		if scalars == nil {
			return ErrFieldTypeNotMatch
		}
		data := scalars.GetIntData()
		f.SetInt(int64(data.Data[idx]))
	case entity.FieldTypeInt64:
		if f.Kind() != reflect.Int64 {
			return ErrFieldTypeNotMatch
		}
		if scalars == nil {
			return ErrFieldTypeNotMatch
		}
		data := scalars.GetIntData()
		f.SetInt(int64(data.Data[idx]))

	case entity.FieldTypeFloat:
		if f.Kind() != reflect.Float32 {
			return ErrFieldTypeNotMatch
		}
		if scalars == nil {
			return ErrFieldTypeNotMatch
		}
		data := scalars.GetFloatData()
		f.SetFloat(float64(data.Data[idx]))
	case entity.FieldTypeDouble:
		if f.Kind() != reflect.Float64 {
			return ErrFieldTypeNotMatch
		}
		if scalars == nil {
			return ErrFieldTypeNotMatch
		}
		data := scalars.GetDoubleData()
		f.SetFloat(data.Data[idx])
	case entity.FieldTypeString:
		if f.Kind() != reflect.String {
			return ErrFieldTypeNotMatch
		}
		if scalars == nil {
			return ErrFieldTypeNotMatch
		}
		data := scalars.GetStringData()
		f.SetString(data.Data[idx])

	case entity.FieldTypeFloatVector:
		if vectors == nil {
			return ErrFieldTypeNotMatch
		}
		data := vectors.GetFloatVector()
		if data == nil {

			return ErrFieldTypeNotMatch
		}
		vector := data.Data[idx*int(vectors.Dim) : (idx+1)*int(vectors.Dim)]
		switch f.Kind() {
		case reflect.Slice:
			f.Set(reflect.ValueOf(vector))
		case reflect.Array:
			arrType := reflect.ArrayOf(int(vectors.Dim), reflect.TypeOf(float32(0)))
			arr := reflect.New(arrType).Elem()
			for i := 0; i < int(vectors.Dim); i++ {
				arr.Index(i).Set(reflect.ValueOf(vector[i]))
			}
			f.Set(arr)
		default:
			return ErrFieldTypeNotMatch
		}
	case entity.FieldTypeBinaryVector:
		if vectors == nil {
			return ErrFieldTypeNotMatch
		}
		data := vectors.GetBinaryVector()
		if data == nil {
			return ErrFieldTypeNotMatch
		}
		vector := data[idx*int(vectors.Dim/8) : (idx+1)*int(vectors.Dim/8)]
		switch f.Kind() {
		case reflect.Slice:
			f.Set(reflect.ValueOf(vector))
		case reflect.Array:
			arrType := reflect.ArrayOf(int(vectors.Dim), reflect.TypeOf(byte(0)))
			arr := reflect.New(arrType).Elem()
			for i := 0; i < int(vectors.Dim/8); i++ {
				arr.Index(i).Set(reflect.ValueOf(vector[i]))
			}
			f.Set(arr)
		default:
			return ErrFieldTypeNotMatch
		}
	}
	return nil
}
