package client

import (
	"context"
	"errors"
	"fmt"
	"reflect"

	"github.com/golang/protobuf/proto"
	server "github.com/milvus-io/milvus-proto/go-api/milvuspb"
	schema "github.com/milvus-io/milvus-proto/go-api/schemapb"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
)

// CreateCollectionByRow create collection by row
func (c *GrpcClient) CreateCollectionByRow(ctx context.Context, row entity.Row, shardNum int32) error {
	if c.Service == nil {
		return ErrClientNotReady
	}
	// parse schema from row definition
	sch, err := entity.ParseSchema(row)
	if err != nil {
		return err
	}

	// check collection already exists
	has, err := c.HasCollection(ctx, sch.CollectionName)
	if err != nil {
		return err
	}
	// already exists collection with same name, return error
	if has {
		return fmt.Errorf("collection %s already exist", sch.CollectionName)
	}
	// marshal schema to bytes for message transfer
	p := sch.ProtoMessage()
	bs, err := proto.Marshal(p)
	if err != nil {
		return err
	}
	// compose request and invoke Service
	req := &server.CreateCollectionRequest{
		DbName:         "", // reserved fields, not used for now
		CollectionName: sch.CollectionName,
		Schema:         bs,
		ShardsNum:      shardNum,
	}
	resp, err := c.Service.CreateCollection(ctx, req)
	// handles response
	if err != nil {
		return err
	}
	err = handleRespStatus(resp)
	if err != nil {
		return nil
	}
	return nil
}

// InsertByRows insert by rows
func (c *GrpcClient) InsertByRows(ctx context.Context, collName string, partitionName string,
	rows []entity.Row) (entity.Column, error) {
	if c.Service == nil {
		return nil, ErrClientNotReady
	}
	if len(rows) == 0 {
		return nil, errors.New("empty rows provided")
	}

	if err := c.checkCollectionExists(ctx, collName); err != nil {
		return nil, err
	}
	coll, err := c.DescribeCollection(ctx, collName)
	if err != nil {
		return nil, err
	}
	// 1. convert rows to columns
	columns, err := entity.RowsToColumns(rows, coll.Schema)
	if err != nil {
		return nil, err
	}
	// 2. do insert request
	req := &server.InsertRequest{
		DbName:         "", // reserved
		CollectionName: collName,
		PartitionName:  partitionName,
	}
	if req.PartitionName == "" {
		req.PartitionName = "_default" // use default partition
	}
	req.NumRows = uint32(len(rows))
	for _, column := range columns {
		req.FieldsData = append(req.FieldsData, column.FieldData())
	}
	resp, err := c.Service.Insert(ctx, req)
	if err != nil {
		return nil, err
	}
	if err := handleRespStatus(resp.GetStatus()); err != nil {
		return nil, err
	}
	MetaCache.setSessionTs(collName, resp.Timestamp)
	// 3. parse id column
	return entity.IDColumns(resp.GetIDs(), 0, -1)
}

// SearchResultByRows search result for row-based Search
type SearchResultByRows struct {
	ResultCount int
	Scores      []float32
	Rows        []entity.Row
	Err         error
}

// SearchResultToRows converts search result proto to rows
func SearchResultToRows(sch *entity.Schema, results *schema.SearchResultData, t reflect.Type, output map[string]struct{}) ([]SearchResultByRows, error) {
	var err error
	offset := 0
	// new will have a pointer, so de-reference first if type is pointer to struct
	if t.Kind() == reflect.Ptr {
		t = t.Elem()
	}
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

			// extract primary field logic
			for _, field := range sch.Fields {
				f := v.FieldByName(field.Name) // TODO silverxia field may be annotated by tags, which means the field name will not be the same
				if !f.IsValid() {
					if _, has := output[field.Name]; has {
						// TODO silverxia in output field list but not defined in rows
					}
					continue
				}
				// Primary key has different field from search result
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
					}
					continue
				}

				// fieldDataList
				fieldData, has := nameFieldData[field.Name]
				if !has || fieldData == nil {
					if _, has := output[field.Name]; has {
						// TODO silverxia in output field list but not defined in rows
					}

					continue
				}

				// Set field value with offset+j-th item
				err = SetFieldValue(field, f, fieldData, offset+j)
				if err != nil {
					entry.Err = err
					break
				}

			}
			r := p.Interface()
			row, ok := r.(entity.Row)
			if ok {
				entry.Rows = append(entry.Rows, row)
			}
		}
		sr = append(sr, entry)
		// set offset after processed one result
		offset += rc
	}
	return sr, nil
}

var (
	// ErrFieldTypeNotMatch error for field type not match
	ErrFieldTypeNotMatch = errors.New("field type not matched")
)

// SetFieldValue set row field value with reflection
func SetFieldValue(field *entity.Field, f reflect.Value, fieldData *schema.FieldData, idx int) error {
	scalars := fieldData.GetScalars()
	vectors := fieldData.GetVectors()
	// This switch part is messy
	// Maybe this can be refactored later
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
		if data == nil {
			return ErrFieldTypeNotMatch
		}
		f.SetInt(int64(data.Data[idx]))
	case entity.FieldTypeInt32:
		if f.Kind() != reflect.Int32 {
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
	case entity.FieldTypeInt64:
		if f.Kind() != reflect.Int64 {
			return ErrFieldTypeNotMatch
		}
		if scalars == nil {
			return ErrFieldTypeNotMatch
		}
		data := scalars.GetLongData()
		if data == nil {
			return ErrFieldTypeNotMatch
		}
		f.SetInt(data.Data[idx])
	case entity.FieldTypeFloat:
		if f.Kind() != reflect.Float32 {
			return ErrFieldTypeNotMatch
		}
		if scalars == nil {
			return ErrFieldTypeNotMatch
		}
		data := scalars.GetFloatData()
		if data == nil {
			return ErrFieldTypeNotMatch
		}

		f.SetFloat(float64(data.Data[idx]))
	case entity.FieldTypeDouble:
		if f.Kind() != reflect.Float64 {
			return ErrFieldTypeNotMatch
		}
		if scalars == nil {
			return ErrFieldTypeNotMatch
		}
		data := scalars.GetDoubleData()
		if data == nil {
			return ErrFieldTypeNotMatch
		}

		f.SetFloat(data.Data[idx])
	case entity.FieldTypeString:
		if f.Kind() != reflect.String {
			return ErrFieldTypeNotMatch
		}
		if scalars == nil {
			return ErrFieldTypeNotMatch
		}
		data := scalars.GetStringData()
		if data == nil {
			return ErrFieldTypeNotMatch
		}

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
			arrType := reflect.ArrayOf(int(vectors.Dim/8), reflect.TypeOf(byte(0)))
			arr := reflect.New(arrType).Elem()
			for i := 0; i < int(vectors.Dim/8); i++ {
				arr.Index(i).Set(reflect.ValueOf(vector[i]))
			}
			f.Set(arr)
		default:
			return ErrFieldTypeNotMatch
		}
	default:
		return ErrFieldTypeNotMatch
	}
	return nil
}
