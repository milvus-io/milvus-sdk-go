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
	"fmt"
	"time"

	"github.com/cockroachdb/errors"
	"github.com/golang/protobuf/proto"
	"github.com/milvus-io/milvus-proto/go-api/v2/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v2/milvuspb"
	"github.com/milvus-io/milvus-proto/go-api/v2/schemapb"

	"github.com/milvus-io/milvus-sdk-go/v2/entity"
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

	// convert columns to field data
	fieldsData, rowSize, err := c.processInsertColumns(coll.Schema, columns...)
	if err != nil {
		return nil, err
	}

	// 2. do insert request
	req := &milvuspb.InsertRequest{
		DbName:         "", // reserved
		CollectionName: collName,
		PartitionName:  partitionName,
		FieldsData:     fieldsData,
	}

	req.NumRows = uint32(rowSize)

	resp, err := c.Service.Insert(ctx, req)
	if err != nil {
		return nil, err
	}
	if err := handleRespStatus(resp.GetStatus()); err != nil {
		return nil, err
	}
	MetaCache.setSessionTs(collName, resp.Timestamp)
	// 3. parse id column
	return entity.IDColumns(coll.Schema, resp.GetIDs(), 0, -1)
}

func (c *GrpcClient) processInsertColumns(colSchema *entity.Schema, columns ...entity.Column) ([]*schemapb.FieldData, int, error) {
	// setup dynamic related var
	isDynamic := colSchema.EnableDynamicField

	// check columns and field matches
	var rowSize int
	mNameField := make(map[string]*entity.Field)
	for _, field := range colSchema.Fields {
		mNameField[field.Name] = field
	}
	mNameColumn := make(map[string]entity.Column)
	var dynamicColumns []entity.Column
	for _, column := range columns {
		_, dup := mNameColumn[column.Name()]
		if dup {
			return nil, 0, fmt.Errorf("duplicated column %s found", column.Name())
		}
		l := column.Len()
		if rowSize == 0 {
			rowSize = l
		} else {
			if rowSize != l {
				return nil, 0, errors.New("column size not match")
			}
		}
		field, has := mNameField[column.Name()]
		if !has {
			if !isDynamic {
				return nil, 0, fmt.Errorf("field %s does not exist in collection %s", column.Name(), colSchema.CollectionName)
			}
			// add to dynamic column list for further processing
			dynamicColumns = append(dynamicColumns, column)
			continue
		}

		mNameColumn[column.Name()] = column
		if column.Type() != field.DataType {
			return nil, 0, fmt.Errorf("param column %s has type %v but collection field definition is %v", column.Name(), column.FieldData(), field.DataType)
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
				return nil, 0, fmt.Errorf("params column %s vector dim %d not match collection definition, which has dim of %s", field.Name, dim, field.TypeParams[entity.TypeParamDim])
			}
		}
	}

	// check all fixed field pass value
	for _, field := range colSchema.Fields {
		_, has := mNameColumn[field.Name]
		if !has &&
			!field.AutoID && !field.IsDynamic {
			return nil, 0, fmt.Errorf("field %s not passed", field.Name)
		}
	}

	fieldsData := make([]*schemapb.FieldData, 0, len(mNameColumn)+1)
	for _, fixedColumn := range mNameColumn {
		fieldsData = append(fieldsData, fixedColumn.FieldData())
	}
	if len(dynamicColumns) > 0 {
		// use empty column name here
		col, err := c.mergeDynamicColumns("", rowSize, dynamicColumns)
		if err != nil {
			return nil, 0, err
		}
		fieldsData = append(fieldsData, col)
	}

	return fieldsData, rowSize, nil
}

func (c *GrpcClient) mergeDynamicColumns(dynamicName string, rowSize int, columns []entity.Column) (*schemapb.FieldData, error) {
	values := make([][]byte, 0, rowSize)
	for i := 0; i < rowSize; i++ {
		m := make(map[string]interface{})
		for _, column := range columns {
			// range guaranteed
			m[column.Name()], _ = column.Get(i)
		}
		bs, err := json.Marshal(m)
		if err != nil {
			return nil, err
		}
		values = append(values, bs)
	}
	return &schemapb.FieldData{
		Type:      schemapb.DataType_JSON,
		FieldName: dynamicName,
		Field: &schemapb.FieldData_Scalars{
			Scalars: &schemapb.ScalarField{
				Data: &schemapb.ScalarField_JsonData{
					JsonData: &schemapb.JSONArray{
						Data: values,
					},
				},
			},
		},
		IsDynamic: true,
	}, nil
}

// Flush force collection to flush memory records into storage
// in sync mode, flush will wait all segments to be flushed
func (c *GrpcClient) Flush(ctx context.Context, collName string, async bool, opts ...FlushOption) error {
	_, _, _, err := c.FlushV2(ctx, collName, async, opts...)
	return err
}

// Flush force collection to flush memory records into storage
// in sync mode, flush will wait all segments to be flushed
func (c *GrpcClient) FlushV2(ctx context.Context, collName string, async bool, opts ...FlushOption) ([]int64, []int64, int64, error) {
	if c.Service == nil {
		return nil, nil, 0, ErrClientNotReady
	}
	if err := c.checkCollectionExists(ctx, collName); err != nil {
		return nil, nil, 0, err
	}
	req := &milvuspb.FlushRequest{
		DbName:          "", // reserved,
		CollectionNames: []string{collName},
	}
	for _, opt := range opts {
		opt(req)
	}
	resp, err := c.Service.Flush(ctx, req)
	if err != nil {
		return nil, nil, 0, err
	}
	if err := handleRespStatus(resp.GetStatus()); err != nil {
		return nil, nil, 0, err
	}
	if !async {
		segmentIDs, has := resp.GetCollSegIDs()[collName]
		ids := segmentIDs.GetData()
		if has && len(ids) > 0 {
			flushed := func() bool {
				resp, err := c.Service.GetFlushState(ctx, &milvuspb.GetFlushStateRequest{
					SegmentIDs:     ids,
					FlushTs:        resp.GetCollFlushTs()[collName],
					CollectionName: collName,
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
					return nil, nil, 0, errors.New("deadline exceeded")
				default:
				}
				time.Sleep(200 * time.Millisecond)
			}
		}
	}
	return resp.GetCollSegIDs()[collName].GetData(), resp.GetFlushCollSegIDs()[collName].GetData(), resp.GetCollSealTimes()[collName], nil
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
	if ids.Name() != "" && pkf.Name != ids.Name() {
		return errors.New("only delete by primary key is supported now")
	}

	expr := PKs2Expr(pkf.Name, ids)

	req := &milvuspb.DeleteRequest{
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

// Delete deletes entries match expression
func (c *GrpcClient) Delete(ctx context.Context, collName string, partitionName string, expr string) error {
	if c.Service == nil {
		return ErrClientNotReady
	}

	// check collection name
	if err := c.checkCollectionExists(ctx, collName); err != nil {
		return err
	}

	// check partition name
	if partitionName != "" {
		err := c.checkPartitionExists(ctx, collName, partitionName)
		if err != nil {
			return err
		}
	}

	req := &milvuspb.DeleteRequest{
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

// Upsert Index into collection with column-based format
// collName is the collection name
// partitionName is the partition to upsert, if not specified(empty), default partition will be used
// columns are slice of the column-based data
func (c *GrpcClient) Upsert(ctx context.Context, collName string, partitionName string, columns ...entity.Column) (entity.Column, error) {
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

	fieldsData, rowSize, err := c.processInsertColumns(coll.Schema, columns...)
	if err != nil {
		return nil, err
	}

	// 2. do upsert request
	req := &milvuspb.UpsertRequest{
		DbName:         "", // reserved
		CollectionName: collName,
		PartitionName:  partitionName,
		FieldsData:     fieldsData,
	}

	req.NumRows = uint32(rowSize)

	resp, err := c.Service.Upsert(ctx, req)
	if err != nil {
		return nil, err
	}
	if err := handleRespStatus(resp.GetStatus()); err != nil {
		return nil, err
	}
	MetaCache.setSessionTs(collName, resp.Timestamp)
	// 3. parse id column
	return entity.IDColumns(coll.Schema, resp.GetIDs(), 0, -1)
}

// BulkInsert data files(json, numpy, etc.) on MinIO/S3 storage, read and parse them into sealed segments
func (c *GrpcClient) BulkInsert(ctx context.Context, collName string, partitionName string, files []string, opts ...BulkInsertOption) (int64, error) {
	if c.Service == nil {
		return 0, ErrClientNotReady
	}
	req := &milvuspb.ImportRequest{
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
	req := &milvuspb.GetImportStateRequest{
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
	req := &milvuspb.ListImportTasksRequest{
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

func vector2PlaceholderGroupBytes(vectors []entity.Vector) []byte {
	phg := &commonpb.PlaceholderGroup{
		Placeholders: []*commonpb.PlaceholderValue{
			vector2Placeholder(vectors),
		},
	}

	bs, _ := proto.Marshal(phg)
	return bs
}

func vector2Placeholder(vectors []entity.Vector) *commonpb.PlaceholderValue {
	var placeHolderType commonpb.PlaceholderType
	ph := &commonpb.PlaceholderValue{
		Tag:    "$0",
		Values: make([][]byte, 0, len(vectors)),
	}
	if len(vectors) == 0 {
		return ph
	}
	switch vectors[0].(type) {
	case entity.FloatVector:
		placeHolderType = commonpb.PlaceholderType_FloatVector
	case entity.BinaryVector:
		placeHolderType = commonpb.PlaceholderType_BinaryVector
	case entity.BFloat16Vector:
		placeHolderType = commonpb.PlaceholderType_BFloat16Vector
	case entity.Float16Vector:
		placeHolderType = commonpb.PlaceholderType_Float16Vector
	case entity.SparseEmbedding:
		placeHolderType = commonpb.PlaceholderType_SparseFloatVector
	}
	ph.Type = placeHolderType
	for _, vector := range vectors {
		ph.Values = append(ph.Values, vector.Serialize())
	}
	return ph
}
