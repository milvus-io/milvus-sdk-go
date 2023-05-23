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

package client

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"strconv"
	"strings"
	"time"

	"github.com/golang/protobuf/proto"
	common "github.com/milvus-io/milvus-proto/go-api/commonpb"
	server "github.com/milvus-io/milvus-proto/go-api/milvuspb"
	schema "github.com/milvus-io/milvus-proto/go-api/schemapb"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
)

const (
	offsetKey        = `offset`
	limitKey         = `limit`
	ignoreGrowingKey = `ignore_growing`
)

// Insert Index  into collection with column-based format
// collName is the collection name
// partitionName is the partition to insert, if not specified(empty), default partition will be used
// columns are slice of the column-based data
func (c *GrpcClient) Insert(ctx context.Context, collName string, partitionName string, columns ...entity.Column) (entity.Column, error) {
	if c.Service == nil {
		return nil, ErrClientNotReady
	}
	// 1. validation for all input params
	// collection
	if err := c.checkCollectionExists(ctx, collName); err != nil {
		return nil, err
	}
	if partitionName != "" {
		err := c.checkPartitionExists(ctx, collName, partitionName)
		if err != nil {
			return nil, err
		}
	}
	// fields
	var rowSize int
	coll, err := c.DescribeCollection(ctx, collName)
	if err != nil {
		return nil, err
	}
	mNameField := make(map[string]*entity.Field)
	for _, field := range coll.Schema.Fields {
		mNameField[field.Name] = field
	}
	mNameColumn := make(map[string]entity.Column)
	for _, column := range columns {
		mNameColumn[column.Name()] = column
		l := column.Len()
		if rowSize == 0 {
			rowSize = l
		} else {
			if rowSize != l {
				return nil, errors.New("column size not match")
			}
		}
		field, has := mNameField[column.Name()]
		if !has {
			return nil, fmt.Errorf("field %s does not exist in collection %s", column.Name(), collName)
		}
		if column.Type() != field.DataType {
			return nil, fmt.Errorf("param column %s has type %v but collection field definition is %v", column.Name(), column.FieldData(), field.DataType)
		}
		if field.DataType == entity.FieldTypeFloatVector || field.DataType == entity.FieldTypeBinaryVector {
			dim := 0
			switch column := column.(type) {
			case *entity.ColumnFloatVector:
				dim = column.Dim()
			case *entity.ColumnBinaryVector:
				dim = column.Dim()
			}
			if fmt.Sprintf("%d", dim) != field.TypeParams[entity.TypeParamDim] {
				return nil, fmt.Errorf("params column %s vector dim %d not match collection definition, which has dim of %s", field.Name, dim, field.TypeParams[entity.TypeParamDim])
			}
		}
	}
	for _, field := range coll.Schema.Fields {
		_, has := mNameColumn[field.Name]
		if !has && !field.AutoID {
			return nil, fmt.Errorf("field %s not passed", field.Name)
		}
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
	req.NumRows = uint32(rowSize)
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

// Flush force collection to flush memory records into storage
// in sync mode, flush will wait all segments to be flushed
func (c *GrpcClient) Flush(ctx context.Context, collName string, async bool) error {
	if c.Service == nil {
		return ErrClientNotReady
	}
	if err := c.checkCollectionExists(ctx, collName); err != nil {
		return err
	}
	req := &server.FlushRequest{
		DbName:          "", // reserved,
		CollectionNames: []string{collName},
	}
	resp, err := c.Service.Flush(ctx, req)
	if err != nil {
		return err
	}
	if err := handleRespStatus(resp.GetStatus()); err != nil {
		return err
	}
	if !async {
		segmentIDs, has := resp.GetCollSegIDs()[collName]
		ids := segmentIDs.GetData()
		if has && len(ids) > 0 {
			flushed := func() bool {
				resp, err := c.Service.GetFlushState(ctx, &server.GetFlushStateRequest{
					SegmentIDs: ids,
				})
				if err != nil {
					// TODO max retry
					return false
				}
				return resp.GetFlushed()
			}
			for !flushed() {
				// respect context deadline/cancel
				select {
				case <-ctx.Done():
					return errors.New("deadline exceeded")
				default:
				}
				time.Sleep(200 * time.Millisecond)
			}
		}
	}
	return nil
}

// DeleteByPks deletes entries related to provided primary keys
func (c *GrpcClient) DeleteByPks(ctx context.Context, collName string, partitionName string, ids entity.Column) error {
	if c.Service == nil {
		return ErrClientNotReady
	}

	// check collection name
	if err := c.checkCollectionExists(ctx, collName); err != nil {
		return err
	}
	coll, err := c.DescribeCollection(ctx, collName)
	if err != nil {
		return err
	}
	// check partition name
	if partitionName != "" {
		err := c.checkPartitionExists(ctx, collName, partitionName)
		if err != nil {
			return err
		}
	}
	// check primary keys
	if ids.Len() == 0 {
		return errors.New("ids len must not be zero")
	}
	if ids.Type() != entity.FieldTypeInt64 && ids.Type() != entity.FieldTypeVarChar { // string key not supported yet
		return errors.New("only int64 and varchar column can be primary key for now")
	}

	pkf := getPKField(coll.Schema)
	// pkf shall not be nil since is returned from milvus
	if pkf.Name != ids.Name() {
		return errors.New("only delete by primary key is supported now")
	}

	expr := PKs2Expr(ids)

	req := &server.DeleteRequest{
		DbName:         "",
		CollectionName: collName,
		PartitionName:  partitionName,
		Expr:           expr,
	}

	resp, err := c.Service.Delete(ctx, req)
	if err != nil {
		return err
	}
	err = handleRespStatus(resp.GetStatus())
	if err != nil {
		return err
	}
	MetaCache.setSessionTs(collName, resp.Timestamp)
	return nil
}

// Search with bool expression
func (c *GrpcClient) Search(ctx context.Context, collName string, partitions []string,
	expr string, outputFields []string, vectors []entity.Vector, vectorField string, metricType entity.MetricType, topK int, sp entity.SearchParam, opts ...SearchQueryOptionFunc) ([]SearchResult, error) {
	if c.Service == nil {
		return []SearchResult{}, ErrClientNotReady
	}
	_, ok := MetaCache.getCollectionInfo(collName)
	if !ok {
		c.DescribeCollection(ctx, collName)
	}
	option, err := makeSearchQueryOption(collName, opts...)
	if err != nil {
		return nil, err
	}
	// 2. Request milvus Service
	req, err := prepareSearchRequest(collName, partitions, expr, outputFields, vectors, vectorField, metricType, topK, sp, option)
	if err != nil {
		return nil, err
	}

	sr := make([]SearchResult, 0, len(vectors))
	resp, err := c.Service.Search(ctx, req)
	if err != nil {
		return nil, err
	}
	if err := handleRespStatus(resp.GetStatus()); err != nil {
		return nil, err
	}
	// 3. parse result into result
	results := resp.GetResults()
	offset := 0
	fieldDataList := results.GetFieldsData()
	for i := 0; i < int(results.GetNumQueries()); i++ {
		rc := int(results.GetTopks()[i]) // result entry count for current query
		entry := SearchResult{
			ResultCount: rc,
			Scores:      results.GetScores()[offset : offset+rc],
		}
		entry.IDs, entry.Err = entity.IDColumns(results.GetIds(), offset, offset+rc)
		if entry.Err != nil {
			offset += rc
			continue
		}
		entry.Fields = make([]entity.Column, 0, len(fieldDataList))
		for _, fieldData := range fieldDataList {
			column, err := entity.FieldDataColumn(fieldData, offset, offset+rc)
			if err != nil {
				entry.Err = err
				continue
			}
			entry.Fields = append(entry.Fields, column)
		}
		sr = append(sr, entry)
		offset += rc
	}
	return sr, nil
}

func PKs2Expr(ids entity.Column) string {
	var expr string
	switch ids.Type() {
	case entity.FieldTypeInt64:
		expr = fmt.Sprintf("%s in %s", ids.Name(), strings.Join(strings.Fields(fmt.Sprint(ids.FieldData().GetScalars().GetLongData().GetData())), ","))
	case entity.FieldTypeVarChar:
		data := ids.FieldData().GetScalars().GetData().(*schema.ScalarField_StringData).StringData.Data
		for i := range data {
			data[i] = fmt.Sprintf("\"%s\"", data[i])
		}
		expr = fmt.Sprintf("%s in %s", ids.Name(), strings.Join(strings.Fields(fmt.Sprint(data)), ","))
	}
	return expr
}

// QueryByPks query record by specified primary key(s)
func (c *GrpcClient) QueryByPks(ctx context.Context, collectionName string, partitionNames []string, ids entity.Column, outputFields []string, opts ...SearchQueryOptionFunc) ([]entity.Column, error) {
	if c.Service == nil {
		return nil, ErrClientNotReady
	}
	// check primary keys
	if ids.Len() == 0 {
		return nil, errors.New("ids len must not be zero")
	}
	if ids.Type() != entity.FieldTypeInt64 && ids.Type() != entity.FieldTypeVarChar { // string key not supported yet
		return nil, errors.New("only int64 and varchar column can be primary key for now")
	}

	expr := PKs2Expr(ids)

	return c.Query(ctx, collectionName, partitionNames, expr, outputFields, opts...)
}

// Query performs query by expression.
func (c *GrpcClient) Query(ctx context.Context, collectionName string, partitionNames []string, expr string, outputFields []string, opts ...SearchQueryOptionFunc) ([]entity.Column, error) {
	if c.Service == nil {
		return nil, ErrClientNotReady
	}

	_, ok := MetaCache.getCollectionInfo(collectionName)
	if !ok {
		_, err := c.DescribeCollection(ctx, collectionName)
		if err != nil {
			return nil, err
		}
	}

	option, err := makeSearchQueryOption(collectionName, opts...)
	if err != nil {
		return nil, err
	}

	req := &server.QueryRequest{
		DbName:             "", // reserved field
		CollectionName:     collectionName,
		Expr:               expr,
		OutputFields:       outputFields,
		PartitionNames:     partitionNames,
		GuaranteeTimestamp: option.GuaranteeTimestamp,
		TravelTimestamp:    option.TravelTimestamp,
	}
	if option.Offset > 0 {
		req.QueryParams = append(req.QueryParams, &common.KeyValuePair{Key: offsetKey, Value: strconv.FormatInt(option.Offset, 10)})
	}
	if option.Limit > 0 {
		req.QueryParams = append(req.QueryParams, &common.KeyValuePair{Key: limitKey, Value: strconv.FormatInt(option.Limit, 10)})
	}
	if option.IgnoreGrowing {
		req.QueryParams = append(req.QueryParams, &common.KeyValuePair{Key: ignoreGrowingKey, Value: strconv.FormatBool(option.IgnoreGrowing)})
	}

	resp, err := c.Service.Query(ctx, req)
	if err != nil {
		return nil, err
	}
	err = handleRespStatus(resp.GetStatus())
	if err != nil {
		return nil, err
	}

	fieldsData := resp.GetFieldsData()
	columns := make([]entity.Column, 0, len(fieldsData))
	for _, fieldData := range resp.GetFieldsData() {
		if fieldData.GetType() == schema.DataType_FloatVector ||
			fieldData.GetType() == schema.DataType_BinaryVector {
			column, err := entity.FieldDataVector(fieldData)
			if err != nil {
				return nil, err
			}
			columns = append(columns, column)
			continue
		}
		column, err := entity.FieldDataColumn(fieldData, 0, -1)
		if err != nil {
			return nil, err
		}
		columns = append(columns, column)
	}

	return columns, nil
}

func getPKField(schema *entity.Schema) *entity.Field {
	for _, f := range schema.Fields {
		if f.PrimaryKey {
			return f
		}
	}
	return nil
}

func prepareSearchRequest(collName string, partitions []string,
	expr string, outputFields []string, vectors []entity.Vector, vectorField string,
	metricType entity.MetricType, topK int, sp entity.SearchParam, opt *SearchQueryOption) (*server.SearchRequest, error) {
	params := sp.Params()
	bs, err := json.Marshal(params)
	if err != nil {
		return nil, err
	}

	searchParams := entity.MapKvPairs(map[string]string{
		"anns_field":     vectorField,
		"topk":           fmt.Sprintf("%d", topK),
		"params":         string(bs),
		"metric_type":    string(metricType),
		"round_decimal":  "-1",
		ignoreGrowingKey: strconv.FormatBool(opt.IgnoreGrowing),
		offsetKey:        fmt.Sprintf("%d", opt.Offset),
	})
	req := &server.SearchRequest{
		DbName:             "",
		CollectionName:     collName,
		PartitionNames:     partitions,
		Dsl:                expr,
		PlaceholderGroup:   vector2PlaceholderGroupBytes(vectors),
		DslType:            common.DslType_BoolExprV1,
		OutputFields:       outputFields,
		SearchParams:       searchParams,
		GuaranteeTimestamp: opt.GuaranteeTimestamp,
		TravelTimestamp:    opt.TravelTimestamp,
		Nq:                 int64(len(vectors)),
	}
	return req, nil
}

// GetPersistentSegmentInfo get persistent segment info
func (c *GrpcClient) GetPersistentSegmentInfo(ctx context.Context, collName string) ([]*entity.Segment, error) {
	if c.Service == nil {
		return []*entity.Segment{}, ErrClientNotReady
	}
	req := &server.GetPersistentSegmentInfoRequest{
		DbName:         "", // reserved
		CollectionName: collName,
	}
	resp, err := c.Service.GetPersistentSegmentInfo(ctx, req)
	if err != nil {
		return []*entity.Segment{}, err
	}
	if err := handleRespStatus(resp.GetStatus()); err != nil {
		return []*entity.Segment{}, err
	}
	segments := make([]*entity.Segment, 0, len(resp.GetInfos()))
	for _, info := range resp.GetInfos() {
		segments = append(segments, &entity.Segment{
			ID:           info.GetSegmentID(),
			CollectionID: info.GetCollectionID(),
			ParititionID: info.GetPartitionID(),
			NumRows:      info.GetNumRows(),
			State:        info.GetState(),
		})
	}

	return segments, nil
}

// GetQuerySegmentInfo get query query cluster segment loaded info
func (c *GrpcClient) GetQuerySegmentInfo(ctx context.Context, collName string) ([]*entity.Segment, error) {
	if c.Service == nil {
		return []*entity.Segment{}, ErrClientNotReady
	}
	req := &server.GetQuerySegmentInfoRequest{
		DbName:         "", // reserved
		CollectionName: collName,
	}
	resp, err := c.Service.GetQuerySegmentInfo(ctx, req)
	if err != nil {
		return []*entity.Segment{}, err
	}
	if err := handleRespStatus(resp.GetStatus()); err != nil {
		return []*entity.Segment{}, err
	}

	segments := make([]*entity.Segment, 0, len(resp.GetInfos()))
	for _, info := range resp.GetInfos() {
		segments = append(segments, &entity.Segment{
			ID:           info.GetSegmentID(),
			CollectionID: info.GetCollectionID(),
			ParititionID: info.GetPartitionID(),
			IndexID:      info.GetIndexID(),
			NumRows:      info.GetNumRows(),
		})
	}

	return segments, nil
}

func (c *GrpcClient) CalcDistance(ctx context.Context, collName string, partitions []string,
	metricType entity.MetricType, opLeft, opRight entity.Column) (entity.Column, error) {
	if c.Service == nil {
		return nil, ErrClientNotReady
	}
	if opLeft == nil || opRight == nil {
		return nil, errors.New("operators cannot be nil")
	}

	// check meta
	if err := c.checkCollectionExists(ctx, collName); err != nil {
		return nil, err
	}
	for _, partition := range partitions {
		if err := c.checkPartitionExists(ctx, collName, partition); err != nil {
			return nil, err
		}
	}
	if err := c.checkCollField(ctx, collName, opLeft.Name()); err != nil {
		return nil, err
	}
	if err := c.checkCollField(ctx, collName, opRight.Name()); err != nil {
		return nil, err
	}

	req := &server.CalcDistanceRequest{
		OpLeft:  columnToVectorsArray(collName, partitions, opLeft),
		OpRight: columnToVectorsArray(collName, partitions, opRight),
		Params: entity.MapKvPairs(map[string]string{
			"metric": string(metricType),
		}),
	}
	if req.OpLeft == nil || req.OpRight == nil {
		return nil, errors.New("invalid operator passed")
	}

	resp, err := c.Service.CalcDistance(ctx, req)
	if err != nil {
		return nil, err
	}
	if err := handleRespStatus(resp.GetStatus()); err != nil {
		return nil, err
	}

	if fd := resp.GetFloatDist(); fd != nil {
		return entity.NewColumnFloat("distance", fd.GetData()), nil
	}
	if id := resp.GetIntDist(); id != nil {
		return entity.NewColumnInt32("distance", id.GetData()), nil
	}

	return nil, errors.New("distance field not supported")
}

func columnToVectorsArray(collName string, partitions []string, column entity.Column) *server.VectorsArray {
	result := &server.VectorsArray{}
	switch column.Type() {
	case entity.FieldTypeInt64: // int64 id
		int64Column, ok := column.(*entity.ColumnInt64)
		if !ok {
			return nil // server shall report error
		}
		ids := &server.VectorIDs{
			CollectionName: collName,
			PartitionNames: partitions,
			FieldName:      column.Name(), // TODO use field name or column name?
			IdArray: &schema.IDs{
				IdField: &schema.IDs_IntId{
					IntId: &schema.LongArray{
						Data: int64Column.Data(),
					},
				},
			},
		}
		result.Array = &server.VectorsArray_IdArray{IdArray: ids}
	case entity.FieldTypeString: // string id
		stringColumn, ok := column.(*entity.ColumnString)
		if !ok {
			return nil
		}
		ids := &server.VectorIDs{
			CollectionName: collName,
			PartitionNames: partitions,
			FieldName:      column.Name(),
			IdArray: &schema.IDs{
				IdField: &schema.IDs_StrId{
					StrId: &schema.StringArray{
						Data: stringColumn.Data(),
					},
				},
			},
		}
		result.Array = &server.VectorsArray_IdArray{IdArray: ids}
	case entity.FieldTypeFloatVector:
		fvColumn, ok := column.(*entity.ColumnFloatVector)
		if !ok {
			return nil
		}
		fvdata := fvColumn.Data()
		data := make([]float32, 0, fvColumn.Len()*fvColumn.Dim())
		for _, row := range fvdata {
			data = append(data, row...)
		}
		result.Array = &server.VectorsArray_DataArray{DataArray: &schema.VectorField{
			Dim: int64(fvColumn.Dim()),
			Data: &schema.VectorField_FloatVector{
				FloatVector: &schema.FloatArray{
					Data: data,
				},
			},
		}}
	case entity.FieldTypeBinaryVector:
		bvColumn, ok := column.(*entity.ColumnBinaryVector)
		if !ok {
			return nil
		}
		bvdata := bvColumn.Data()
		data := make([]byte, 0, bvColumn.Dim()*bvColumn.Len()/8)
		for _, row := range bvdata {
			data = append(data, row...)
		}
		result.Array = &server.VectorsArray_DataArray{DataArray: &schema.VectorField{
			Dim: int64(bvColumn.Dim()),
			Data: &schema.VectorField_BinaryVector{
				BinaryVector: data,
			},
		}}
	default:
		return nil
	}
	return result
}

func vector2PlaceholderGroupBytes(vectors []entity.Vector) []byte {
	phg := &common.PlaceholderGroup{
		Placeholders: []*common.PlaceholderValue{
			vector2Placeholder(vectors),
		},
	}

	bs, _ := proto.Marshal(phg)
	return bs
}

func vector2Placeholder(vectors []entity.Vector) *common.PlaceholderValue {
	var placeHolderType common.PlaceholderType
	ph := &common.PlaceholderValue{
		Tag:    "$0",
		Values: make([][]byte, 0, len(vectors)),
	}
	if len(vectors) == 0 {
		return ph
	}
	switch vectors[0].(type) {
	case entity.FloatVector:
		placeHolderType = common.PlaceholderType_FloatVector
	case entity.BinaryVector:
		placeHolderType = common.PlaceholderType_BinaryVector
	}
	ph.Type = placeHolderType
	for _, vector := range vectors {
		ph.Values = append(ph.Values, vector.Serialize())
	}
	return ph
}

func isCollectionPrimaryKey(coll *entity.Collection, column entity.Column) bool {
	if coll == nil || coll.Schema == nil || column == nil {
		return false
	}

	// temporary check logic, since only one primary field is supported
	for _, field := range coll.Schema.Fields {
		if field.PrimaryKey {
			if field.Name == column.Name() && field.DataType == column.Type() {
				return true
			}
			return false
		}
	}
	return false
}

// estRowSize estimate size per row for the specified schema
func estRowSize(sch *entity.Schema, selected []string) int64 {
	var total int64
	for _, field := range sch.Fields {
		if len(selected) > 0 {
			found := false
			for _, sel := range selected {
				if field.Name == sel {
					found = true
					break
				}
			}
			if !found {
				continue
			}
		}
		switch field.DataType {
		case entity.FieldTypeBool:
			total++
		case entity.FieldTypeInt8:
			total++
		case entity.FieldTypeInt16:
			total += 2
		case entity.FieldTypeInt32:
			total += 4
		case entity.FieldTypeInt64:
			total += 8
		case entity.FieldTypeFloat:
			total += 4
		case entity.FieldTypeDouble:
			total += 8
		case entity.FieldTypeString:
			// TODO string need varchar[max] syntax like limitation
		case entity.FieldTypeVarChar:
			maxLength, err := strconv.Atoi(field.TypeParams[entity.TypeParamMaxLength])
			if err != nil {
				log.Fatalf("got invalid varchar max length = %s", field.TypeParams[entity.TypeParamMaxLength])
			}
			total += int64(maxLength)
		case entity.FieldTypeFloatVector:
			dimStr := field.TypeParams[entity.TypeParamDim]
			dim, _ := strconv.ParseInt(dimStr, 10, 64)
			total += 4 * dim
		case entity.FieldTypeBinaryVector:
			dimStr := field.TypeParams[entity.TypeParamDim]
			dim, _ := strconv.ParseInt(dimStr, 10, 64)
			total += 4 * dim / 8
		}
	}
	return total
}

// BulkInsert data files(json, numpy, etc.) on MinIO/S3 storage, read and parse them into sealed segments
func (c *GrpcClient) BulkInsert(ctx context.Context, collName string, partitionName string, files []string, opts ...BulkInsertOption) (int64, error) {
	if c.Service == nil {
		return 0, ErrClientNotReady
	}
	req := &server.ImportRequest{
		CollectionName: collName,
		PartitionName:  partitionName,
		Files:          files,
	}

	for _, opt := range opts {
		opt(req)
	}

	resp, err := c.Service.Import(ctx, req)
	if err != nil {
		return 0, err
	}
	if err := handleRespStatus(resp.GetStatus()); err != nil {
		return 0, err
	}

	return resp.Tasks[0], nil
}

// GetBulkInsertState checks import task state
func (c *GrpcClient) GetBulkInsertState(ctx context.Context, taskID int64) (*entity.BulkInsertTaskState, error) {
	if c.Service == nil {
		return nil, ErrClientNotReady
	}
	req := &server.GetImportStateRequest{
		Task: taskID,
	}
	resp, err := c.Service.GetImportState(ctx, req)
	if err != nil {
		return nil, err
	}
	if err := handleRespStatus(resp.GetStatus()); err != nil {
		return nil, err
	}

	return &entity.BulkInsertTaskState{
		ID:           resp.GetId(),
		State:        entity.BulkInsertState(resp.GetState()),
		RowCount:     resp.GetRowCount(),
		IDList:       resp.GetIdList(),
		Infos:        entity.KvPairsMap(resp.GetInfos()),
		CollectionID: resp.GetCollectionId(),
		SegmentIDs:   resp.GetSegmentIds(),
		CreateTs:     resp.GetCreateTs(),
	}, nil
}

// ListBulkInsertTasks list state of all import tasks
func (c *GrpcClient) ListBulkInsertTasks(ctx context.Context, collName string, limit int64) ([]*entity.BulkInsertTaskState, error) {
	if c.Service == nil {
		return nil, ErrClientNotReady
	}
	req := &server.ListImportTasksRequest{
		CollectionName: collName,
		Limit:          limit,
	}
	resp, err := c.Service.ListImportTasks(ctx, req)
	if err != nil {
		return nil, err
	}
	if err := handleRespStatus(resp.GetStatus()); err != nil {
		return nil, err
	}

	tasks := make([]*entity.BulkInsertTaskState, 0)
	for _, task := range resp.GetTasks() {
		tasks = append(tasks, &entity.BulkInsertTaskState{
			ID:           task.GetId(),
			State:        entity.BulkInsertState(task.GetState()),
			RowCount:     task.GetRowCount(),
			IDList:       task.GetIdList(),
			Infos:        entity.KvPairsMap(task.GetInfos()),
			CollectionID: task.GetCollectionId(),
			SegmentIDs:   task.GetSegmentIds(),
			CreateTs:     task.GetCreateTs(),
		})
	}

	return tasks, nil
}
