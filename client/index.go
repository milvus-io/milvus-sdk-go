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
	"strconv"
	"time"

	"github.com/milvus-io/milvus-proto/go-api/v2/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v2/milvuspb"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
)

const (
	mmapKey = "mmap.enabled"
)

func (c *GrpcClient) checkCollField(ctx context.Context, collName string, fieldName string, filters ...func(string, string, *entity.Field) error) error {
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
			for _, filter := range filters {
				if err := filter(collName, fieldName, f); err != nil {
					return err
				}
			}
			break
		}
	}
	if f == nil {
		return fmt.Errorf("field %s of collection %s does not exist", fieldName, collName)
	}
	return nil
}

func isVectorField(collName, fieldName string, f *entity.Field) error {
	if f.DataType != entity.FieldTypeFloatVector && f.DataType != entity.FieldTypeBinaryVector {
		return fmt.Errorf("field %s of collection %s is not vector field", fieldName, collName)
	}
	return nil
}

type indexDef struct {
	name           string
	fieldName      string
	collectionName string
	params         []*commonpb.KeyValuePair
	MsgBase        *commonpb.MsgBase
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

func WithIndexMsgBase(msgBase *commonpb.MsgBase) IndexOption {
	return func(def *indexDef) {
		def.MsgBase = msgBase
	}
}

func WithMmap(enabled bool) IndexOption {
	return func(id *indexDef) {
		id.params = append(id.params, &commonpb.KeyValuePair{
			Key:   mmapKey,
			Value: strconv.FormatBool(enabled),
		})
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

	idxDef := getIndexDef(opts...)

	req := &milvuspb.CreateIndexRequest{
		Base:           idxDef.MsgBase,
		DbName:         "", // reserved
		CollectionName: collName,
		FieldName:      fieldName,
		IndexName:      idxDef.name,
		ExtraParams:    entity.MapKvPairs(idx.Params()),
	}

	req.ExtraParams = append(req.ExtraParams, idxDef.params...)

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
					case commonpb.IndexState_Finished:
						return nil
					case commonpb.IndexState_Failed:
						return fmt.Errorf("create index failed, reason: %s", desc.GetIndexStateFailReason())
					}
				}
			}

			time.Sleep(100 * time.Millisecond) // wait 100ms
		}
	}
	return nil
}

// AlterIndex modifies the index params
func (c *GrpcClient) AlterIndex(ctx context.Context, collName string, indexName string, opts ...IndexOption) error {
	if c.Service == nil {
		return ErrClientNotReady
	}

	idxDef := getIndexDef(opts...)

	req := &milvuspb.AlterIndexRequest{
		Base:           idxDef.MsgBase,
		DbName:         "", // reserved
		CollectionName: collName,
		IndexName:      indexName,
		ExtraParams:    idxDef.params,
	}

	resp, err := c.Service.AlterIndex(ctx, req)
	if err != nil {
		return err
	}
	return handleRespStatus(resp)
}

// DescribeIndex describe index
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
		if fieldName != "" && info.GetFieldName() != fieldName {
			continue
		}
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

	idxDef := getIndexDef(opts...)
	req := &milvuspb.DropIndexRequest{
		Base:           idxDef.MsgBase,
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
func (c *GrpcClient) GetIndexState(ctx context.Context, collName string, fieldName string, opts ...IndexOption) (entity.IndexState, error) {
	if c.Service == nil {
		return entity.IndexState(commonpb.IndexState_Failed), ErrClientNotReady
	}
	if err := c.checkCollField(ctx, collName, fieldName); err != nil {
		return entity.IndexState(commonpb.IndexState_IndexStateNone), err
	}

	idxDef := getIndexDef(opts...)
	req := &milvuspb.GetIndexStateRequest{
		DbName:         "",
		CollectionName: collName,
		FieldName:      fieldName,
		IndexName:      idxDef.name,
	}
	resp, err := c.Service.GetIndexState(ctx, req)
	if err != nil {
		return entity.IndexState(commonpb.IndexState_IndexStateNone), err
	}
	if err := handleRespStatus(resp.GetStatus()); err != nil {
		return entity.IndexState(commonpb.IndexState_IndexStateNone), err
	}

	return entity.IndexState(resp.GetState()), nil
}

// GetIndexBuildProgress get index building progress
func (c *GrpcClient) GetIndexBuildProgress(ctx context.Context, collName string, fieldName string, opts ...IndexOption) (total, indexed int64, err error) {
	if c.Service == nil {
		return 0, 0, ErrClientNotReady
	}
	if err := c.checkCollField(ctx, collName, fieldName); err != nil {
		return 0, 0, err
	}

	idxDef := getIndexDef(opts...)
	req := &milvuspb.GetIndexBuildProgressRequest{
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

func (c *GrpcClient) describeIndex(ctx context.Context, collName string, fieldName string, opts ...IndexOption) ([]*milvuspb.IndexDescription, error) {
	idxDef := getIndexDef(opts...)
	req := &milvuspb.DescribeIndexRequest{
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
