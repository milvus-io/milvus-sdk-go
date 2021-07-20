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
	"time"

	"github.com/golang/protobuf/proto"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
	"github.com/milvus-io/milvus-sdk-go/v2/internal/proto/common"
	"github.com/milvus-io/milvus-sdk-go/v2/internal/proto/schema"
	"github.com/milvus-io/milvus-sdk-go/v2/internal/proto/server"
)

// Index insert into collection with column-based format
// collName is the collection name
// partitionName is the partition to insert, if not specified(empty), default partition will be used
// columns are slice of the column-based data
func (c *grpcClient) Insert(ctx context.Context, collName string, partitionName string, columns ...entity.Column) (entity.Column, error) {
	if c.service == nil {
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
			if fmt.Sprintf("%d", dim) != field.TypeParams["dim"] {
				return nil, fmt.Errorf("params column %s vector dim %d not match collection definition, which has dim of %s", field.Name, dim, field.TypeParams["dim"])
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
	resp, err := c.service.Insert(ctx, req)
	if err != nil {
		return nil, err
	}
	if err := handleRespStatus(resp.GetStatus()); err != nil {
		return nil, err
	}
	// 3. parse id column
	return entity.IDColumns(resp.GetIDs(), 0, -1)
}

// Flush force collection to flush memory records into storage
// in sync mode, flush will wait all segments to be flushed
func (c *grpcClient) Flush(ctx context.Context, collName string, async bool) error {
	if c.service == nil {
		return ErrClientNotReady
	}
	if err := c.checkCollectionExists(ctx, collName); err != nil {
		return err
	}
	req := &server.FlushRequest{
		DbName:          "", //reserved,
		CollectionNames: []string{collName},
	}
	resp, err := c.service.Flush(ctx, req)
	if err != nil {
		return err
	}
	if err := handleRespStatus(resp.GetStatus()); err != nil {
		return err
	}
	//TODO silverxia, add context done respect logic while waiting
	if !async {
		segmentIDs, has := resp.GetCollSegIDs()[collName]
		if has {
			waitingSet := make(map[int64]struct{})
			for _, segmentID := range segmentIDs.GetData() {
				waitingSet[segmentID] = struct{}{}
			}
			flushed := func() bool {
				segments, err := c.GetPersistentSegmentInfo(context.Background(), collName)
				if err != nil {
					//TODO handles grpc failure, maybe need reconnect?
				}
				flushed := 0
				for _, segment := range segments {
					if _, has := waitingSet[segment.ID]; !has {
						continue
					}
					if !segment.Flushed() {
						return false
					}
					flushed++
				}
				return len(waitingSet) == flushed
			}
			for !flushed() {
				time.Sleep(500 * time.Millisecond)
			}
		}
	}
	return nil
}

//BoolExprSearch search with bool expression
func (c *grpcClient) Search(ctx context.Context, collName string, partitions []string,
	expr string, outputFields []string, vectors []entity.Vector, vectorField string, metricType entity.MetricType, topK int, sp entity.SearchParam) ([]SearchResult, error) {
	if c.service == nil {
		return []SearchResult{}, ErrClientNotReady
	}
	// 1. check all input params
	if err := c.checkCollectionExists(ctx, collName); err != nil {
		return nil, err
	}
	for _, partition := range partitions {
		err := c.checkPartitionExists(ctx, collName, partition)
		if err != nil {
			return nil, err
		}
	}
	// TODO maybe add expr analysis?
	coll, err := c.DescribeCollection(ctx, collName)
	if err != nil {
		return nil, err
	}
	mNameField := make(map[string]*entity.Field)
	for _, field := range coll.Schema.Fields {
		mNameField[field.Name] = field
	}
	for _, outField := range outputFields {
		_, has := mNameField[outField]
		if !has {
			return nil, fmt.Errorf("field %s does not exist in collection %s", outField, collName)
		}
	}
	vfDef, has := mNameField[vectorField]
	if !has {
		return nil, fmt.Errorf("vector field %s does not exist in collection %s", vectorField, collName)
	}
	dimStr := vfDef.TypeParams["dim"]
	for _, vector := range vectors {
		if fmt.Sprintf("%d", vector.Dim()) != dimStr {
			return nil, fmt.Errorf("vector %s has dim of %s while found search vector with dim %d", vectorField,
				dimStr, vector.Dim())
		}
	}
	switch vfDef.DataType {
	case entity.FieldTypeFloatVector:
		if metricType != entity.IP && metricType != entity.L2 {
			return nil, fmt.Errorf("Float vector does not support metric type %s", metricType)
		}
	case entity.FieldTypeBinaryVector:
		if metricType == entity.IP || metricType == entity.L2 {
			return nil, fmt.Errorf("Binary vector does not support metric type %s", metricType)
		}
	}

	// 2. Request milvus service
	bs, _ := json.Marshal(sp.Params())
	searchParams := entity.MapKvPairs(map[string]string{
		"anns_field":  vectorField,
		"topk":        fmt.Sprintf("%d", topK),
		"params":      string(bs),
		"metric_type": string(metricType),
	})

	req := &server.SearchRequest{
		DbName:           "",
		CollectionName:   collName,
		PartitionNames:   []string{},
		SearchParams:     searchParams,
		Dsl:              expr,
		DslType:          common.DslType_BoolExprV1,
		OutputFields:     outputFields,
		PlaceholderGroup: vector2PlaceholderGroupBytes(vectors),
	}
	resp, err := c.service.Search(ctx, req)
	if err != nil {
		return []SearchResult{}, err
	}
	if err := handleRespStatus(resp.GetStatus()); err != nil {
		return []SearchResult{}, err
	}
	// 3. parse result into result
	results := resp.GetResults()
	offset := 0
	sr := make([]SearchResult, 0, results.GetNumQueries())
	fieldDataList := results.GetFieldsData()
	for i := 0; i < int(results.GetNumQueries()); i++ {

		rc := int(results.GetTopks()[i]) // result entry count for current query
		entry := SearchResult{
			ResultCount: rc,
			Scores:      results.GetScores()[offset:rc],
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

// GetPersistentSegmentInfo get persistent segment info
func (c *grpcClient) GetPersistentSegmentInfo(ctx context.Context, collName string) ([]*entity.Segment, error) {
	if c.service == nil {
		return []*entity.Segment{}, ErrClientNotReady
	}
	req := &server.GetPersistentSegmentInfoRequest{
		DbName:         "", //reserved
		CollectionName: collName,
	}
	resp, err := c.service.GetPersistentSegmentInfo(ctx, req)
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
func (c *grpcClient) GetQuerySegmentInfo(ctx context.Context, collName string) ([]*entity.Segment, error) {
	if c.service == nil {
		return []*entity.Segment{}, ErrClientNotReady
	}
	req := &server.GetQuerySegmentInfoRequest{
		DbName:         "", //reserved
		CollectionName: collName,
	}
	resp, err := c.service.GetQuerySegmentInfo(ctx, req)
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

func (c *grpcClient) CalcDistance(ctx context.Context, collName string, partitions []string,
	metricType entity.MetricType, opLeft, opRight entity.Column) (entity.Column, error) {
	if c.service == nil {
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

	resp, err := c.service.CalcDistance(ctx, req)
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
			FieldName:      column.Name(), //TODO use field name or column name?
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
	phg := &server.PlaceholderGroup{
		Placeholders: []*server.PlaceholderValue{
			vector2Placeholder(vectors),
		},
	}

	bs, _ := proto.Marshal(phg)
	return bs
}

func vector2Placeholder(vectors []entity.Vector) *server.PlaceholderValue {
	ph := &server.PlaceholderValue{
		Tag:    "$0",
		Type:   server.PlaceholderType_FloatVector,
		Values: make([][]byte, 0, len(vectors)),
	}
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
