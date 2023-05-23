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
	"fmt"
	"time"

	common "github.com/milvus-io/milvus-proto/go-api/commonpb"
	server "github.com/milvus-io/milvus-proto/go-api/milvuspb"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
)

func (c *GrpcClient) checkCollField(ctx context.Context, collName string, fieldName string) error {
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

type indexDef struct {
	name           string
	fieldName      string
	collectionName string
}

// IndexOption is the predefined function to alter index def.
// shared among create, describe, drop indexes operations.
type IndexOption func(*indexDef)

// WithIndexName returns an IndexOption with customized index name.
func WithIndexName(name string) IndexOption {
	return func(def *indexDef) {
		def.name = name
	}
}

func getIndexDef(opts ...IndexOption) indexDef {
	idxDef := indexDef{}
	for _, opt := range opts {
		opt(&idxDef)
	}
	return idxDef
}

// CreateIndex create index for collection
// Deprecated please use CreateIndexV2 instead.
func (c *GrpcClient) CreateIndex(ctx context.Context, collName string, fieldName string,
	idx entity.Index, async bool, opts ...IndexOption) error {
	if c.Service == nil {
		return ErrClientNotReady
	}
	if err := c.checkCollField(ctx, collName, fieldName); err != nil {
		return err
	}

	flushErr := c.Flush(ctx, collName, true)
	if flushErr != nil {
		return flushErr
	}

	idxDef := getIndexDef(opts...)

	req := &server.CreateIndexRequest{
		DbName:         "", // reserved
		CollectionName: collName,
		FieldName:      fieldName,
		IndexName:      idxDef.name,
		ExtraParams:    entity.MapKvPairs(idx.Params()),
	}

	resp, err := c.Service.CreateIndex(ctx, req)
	if err != nil {
		return err
	}
	if err = handleRespStatus(resp); err != nil {
		return err
	}
	if !async { // sync mode, wait index building result
		for {
			idxDesc, err := c.describeIndex(ctx, collName, fieldName, opts...)
			if err != nil {
				return err
			}
			for _, desc := range idxDesc {
				if (idxDef.name == "" && desc.GetFieldName() == fieldName) || idxDef.name == desc.GetIndexName() {
					switch desc.GetState() {
					case common.IndexState_Finished:
						return nil
					case common.IndexState_Failed:
						return fmt.Errorf("create index failed, reason: %s", desc.GetIndexStateFailReason())
					}
				}
			}

			time.Sleep(100 * time.Millisecond) // wait 100ms
		}
	}
	return nil
}

// DescribeIndex describe index
// Deprecate please use DescribeIndexV2 instead.
func (c *GrpcClient) DescribeIndex(ctx context.Context, collName string, fieldName string, opts ...IndexOption) ([]entity.Index, error) {
	if c.Service == nil {
		return []entity.Index{}, ErrClientNotReady
	}
	if err := c.checkCollField(ctx, collName, fieldName); err != nil {
		return []entity.Index{}, err
	}

	idxDesc, err := c.describeIndex(ctx, collName, fieldName, opts...)
	if err != nil {
		return nil, err
	}

	indexes := make([]entity.Index, 0, len(idxDesc))
	for _, info := range idxDesc {
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
// Deprecate please use DropIndexV2 instead.
func (c *GrpcClient) DropIndex(ctx context.Context, collName string, fieldName string, opts ...IndexOption) error {
	if c.Service == nil {
		return ErrClientNotReady
	}
	if err := c.checkCollField(ctx, collName, fieldName); err != nil {
		return err
	}

	idxDef := getIndexDef(opts...)
	req := &server.DropIndexRequest{
		DbName:         "", //reserved,
		CollectionName: collName,
		FieldName:      fieldName,
		IndexName:      idxDef.name,
	}
	if idxDef.name != "" {
		req.IndexName = idxDef.name
	}

	resp, err := c.Service.DropIndex(ctx, req)
	if err != nil {
		return err
	}
	return handleRespStatus(resp)
}

// GetIndexState get index state
// Deprecate please use DescribeIndexV2 instead.
func (c *GrpcClient) GetIndexState(ctx context.Context, collName string, fieldName string, opts ...IndexOption) (entity.IndexState, error) {
	if c.Service == nil {
		return entity.IndexState(common.IndexState_Failed), ErrClientNotReady
	}
	if err := c.checkCollField(ctx, collName, fieldName); err != nil {
		return entity.IndexState(common.IndexState_IndexStateNone), err
	}

	idxDef := getIndexDef(opts...)
	req := &server.GetIndexStateRequest{
		DbName:         "",
		CollectionName: collName,
		FieldName:      fieldName,
		IndexName:      idxDef.name,
	}
	resp, err := c.Service.GetIndexState(ctx, req)
	if err != nil {
		return entity.IndexState(common.IndexState_IndexStateNone), err
	}
	if err := handleRespStatus(resp.GetStatus()); err != nil {
		return entity.IndexState(common.IndexState_IndexStateNone), err
	}

	return entity.IndexState(resp.GetState()), nil
}

// GetIndexBuildProgress get index building progress
// Deprecate please use DescribeIndexV2 instead.
func (c *GrpcClient) GetIndexBuildProgress(ctx context.Context, collName string, fieldName string, opts ...IndexOption) (total, indexed int64, err error) {
	if c.Service == nil {
		return 0, 0, ErrClientNotReady
	}
	if err := c.checkCollField(ctx, collName, fieldName); err != nil {
		return 0, 0, err
	}

	idxDef := getIndexDef(opts...)
	req := &server.GetIndexBuildProgressRequest{
		DbName:         "",
		CollectionName: collName,
		FieldName:      fieldName,
		IndexName:      idxDef.name,
	}
	resp, err := c.Service.GetIndexBuildProgress(ctx, req)
	if err != nil {
		return 0, 0, err
	}
	if err = handleRespStatus(resp.GetStatus()); err != nil {
		return 0, 0, err
	}
	return resp.GetTotalRows(), resp.GetIndexedRows(), nil
}

func (c *GrpcClient) describeIndex(ctx context.Context, collName string, fieldName string, opts ...IndexOption) ([]*server.IndexDescription, error) {
	idxDef := getIndexDef(opts...)
	req := &server.DescribeIndexRequest{
		CollectionName: collName,
		FieldName:      fieldName,
		IndexName:      idxDef.name,
	}

	resp, err := c.Service.DescribeIndex(ctx, req)
	if err != nil {
		return nil, err
	}
	if err := handleRespStatus(resp.GetStatus()); err != nil {
		return nil, err
	}

	return resp.GetIndexDescriptions(), nil
}
