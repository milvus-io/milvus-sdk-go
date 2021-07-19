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
	"errors"
	"fmt"
	"time"

	"github.com/milvus-io/milvus-sdk-go/v2/entity"
	"github.com/milvus-io/milvus-sdk-go/v2/internal/proto/common"
	"github.com/milvus-io/milvus-sdk-go/v2/internal/proto/server"
)

func (c *grpcClient) checkCollField(ctx context.Context, collName string, fieldName string) error {
	if err := c.checkCollectionExists(ctx, collName); err != nil {
		return err
	}
	coll, err := c.DescribeCollection(ctx, collName)
	if err != nil {
		return err
	}
	var f *entity.Field
	for _, field := range coll.Schema.Fields {
		if field.Name == fieldName {
			f = field
			if f.DataType != entity.FieldTypeFloatVector && f.DataType != entity.FieldTypeBinaryVector {
				return fmt.Errorf("field %s of collection %s is not vector field", fieldName, collName)
			}
			break
		}
	}
	if f == nil {
		return fmt.Errorf("field %s of collection %s does not exist", fieldName, collName)
	}
	return nil
}

// CreateIndex create index for collection
func (c *grpcClient) CreateIndex(ctx context.Context, collName string, fieldName string,
	idx entity.Index, async bool) error {
	if c.service == nil {
		return ErrClientNotReady
	}
	if err := c.checkCollField(ctx, collName, fieldName); err != nil {
		return err
	}

	req := &server.CreateIndexRequest{
		DbName:         "", // reserved
		CollectionName: collName,
		FieldName:      fieldName,
		ExtraParams:    entity.MapKvPairs(idx.Params()),
	}
	resp, err := c.service.CreateIndex(ctx, req)
	if err != nil {
		return err
	}
	if err = handleRespStatus(resp); err != nil {
		return err
	}
	if !async { // sync mode, wait index building result
		for {
			is, err := c.GetIndexState(ctx, collName, fieldName)
			if err != nil {
				return err
			}
			switch is {
			case entity.IndexState(common.IndexState_Failed):
				return errors.New("index build failed")
			case entity.IndexState(common.IndexState_Finished):
				return nil
			default:
			}
			time.Sleep(100 * time.Millisecond) // wait 100ms
		}
	}
	return nil
}

// DescribeIndex describe index
func (c *grpcClient) DescribeIndex(ctx context.Context, collName string, fieldName string) ([]entity.Index, error) {
	if c.service == nil {
		return []entity.Index{}, ErrClientNotReady
	}
	if err := c.checkCollField(ctx, collName, fieldName); err != nil {
		return []entity.Index{}, err
	}
	req := &server.DescribeIndexRequest{
		CollectionName: collName,
		FieldName:      fieldName,
		IndexName:      "", // empty string stands for all index on collection
	}
	resp, err := c.service.DescribeIndex(ctx, req)
	if err != nil {
		return []entity.Index{}, err
	}
	if err := handleRespStatus(resp.GetStatus()); err != nil {
		return []entity.Index{}, err
	}
	indexes := make([]entity.Index, 0, len(resp.GetIndexDescriptions()))
	for _, info := range resp.GetIndexDescriptions() {
		params := entity.KvPairsMap(info.Params)
		it := params["index_type"] // TODO change to const
		idx := entity.NewGenericIndex(
			info.IndexName,
			entity.IndexType(it),
			params,
		)
		indexes = append(indexes, idx)
	}
	return indexes, nil
}

// DropIndex drop index from collection
func (c *grpcClient) DropIndex(ctx context.Context, collName string, fieldName string) error {
	if c.service == nil {
		return ErrClientNotReady
	}
	if err := c.checkCollField(ctx, collName, fieldName); err != nil {
		return err
	}
	req := &server.DropIndexRequest{
		DbName:         "", //reserved,
		CollectionName: collName,
		FieldName:      fieldName,
		IndexName:      "", //reserved
	}
	resp, err := c.service.DropIndex(ctx, req)
	if err != nil {
		return err
	}
	return handleRespStatus(resp)
}

// GetIndexState get index state
func (c *grpcClient) GetIndexState(ctx context.Context, collName string, fieldName string) (entity.IndexState, error) {
	if c.service == nil {
		return entity.IndexState(common.IndexState_Failed), ErrClientNotReady
	}
	if err := c.checkCollField(ctx, collName, fieldName); err != nil {
		return entity.IndexState(common.IndexState_IndexStateNone), err
	}
	req := &server.GetIndexStateRequest{
		DbName:         "",
		CollectionName: collName,
		FieldName:      fieldName,
		IndexName:      "",
	}
	resp, err := c.service.GetIndexState(ctx, req)
	if err != nil {
		return entity.IndexState(common.IndexState_IndexStateNone), err
	}
	if err := handleRespStatus(resp.GetStatus()); err != nil {
		return entity.IndexState(common.IndexState_IndexStateNone), err
	}

	return entity.IndexState(resp.GetState()), nil
}

// GetIndexBuildProgress get index building progress
func (c *grpcClient) GetIndexBuildProgress(ctx context.Context, collName string, fieldName string) (total, indexed int64, err error) {
	if c.service == nil {
		return 0, 0, ErrClientNotReady
	}
	if err := c.checkCollField(ctx, collName, fieldName); err != nil {
		return 0, 0, err
	}
	req := &server.GetIndexBuildProgressRequest{
		DbName:         "",
		CollectionName: collName,
		FieldName:      fieldName,
		IndexName:      "",
	}
	resp, err := c.service.GetIndexBuildProgress(ctx, req)
	if err != nil {
		return 0, 0, err
	}
	if err = handleRespStatus(resp.GetStatus()); err != nil {
		return 0, 0, err
	}
	return resp.GetTotalRows(), resp.GetIndexedRows(), nil
}
