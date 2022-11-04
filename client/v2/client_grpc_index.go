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

package v2

import (
	"context"
	"errors"
	"time"

	common "github.com/milvus-io/milvus-proto/go-api/commonpb"
	server "github.com/milvus-io/milvus-proto/go-api/milvuspb"
	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
)

// handles response status
// if status is nil returns ErrStatusNil
// if status.ErrorCode is common.ErrorCode_Success, returns nil
// otherwise, try use Reason into ErrServiceFailed
// if Reason is empty, returns ErrServiceFailed with default string
func handleRespStatus(status *common.Status) error {
	if status == nil {
		return client.ErrStatusNil
	}
	if status.ErrorCode != common.ErrorCode_Success {
		if status.GetReason() != "" {
			return client.ErrServiceFailed(errors.New(status.GetReason()))
		}
		return client.ErrServiceFailed(errors.New("service failed"))
	}
	return nil
}

type CreateIndexRequestOption func(req *server.CreateIndexRequest)

func SetCollectionNameForCreateIndex(name string) CreateIndexRequestOption {
	return func(req *server.CreateIndexRequest) {
		req.CollectionName = name
	}
}

func SetIndexNameForCreateIndex(name string) CreateIndexRequestOption {
	return func(req *server.CreateIndexRequest) {
		req.IndexName = name
	}
}

func SetFieldNameForCreateIndex(name string) CreateIndexRequestOption {
	return func(req *server.CreateIndexRequest) {
		req.FieldName = name
	}
}

// CreateIndex create index for collection
func (c *grpcClient) CreateIndex(ctx context.Context, async bool, idx entity.Index, opts ...CreateIndexRequestOption) error {
	if c.Service == nil {
		return client.ErrClientNotReady
	}
	req := &server.CreateIndexRequest{
		ExtraParams: entity.MapKvPairs(idx.Params()),
	}
	for _, opt := range opts {
		opt(req)
	}
	if req.GetCollectionName() == "" {
		return errors.New("collection name can not be empty")
	}
	if req.GetFieldName() == "" {
		return errors.New("field name can not be empty")
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
			resp, err := c.Service.DescribeIndex(ctx, &server.DescribeIndexRequest{
				DbName:         "",
				CollectionName: req.GetCollectionName(),
				FieldName:      req.GetFieldName(),
				IndexName:      req.GetIndexName(),
			})
			if err != nil {
				return err
			}
			if err = handleRespStatus(resp.Status); err != nil {
				return err
			}
			for _, index := range resp.GetIndexDescriptions() {
				if (req.GetIndexName() == "" && index.FieldName == req.GetFieldName()) || req.GetIndexName() == index.GetIndexName() {
					switch index.GetState() {
					case common.IndexState_Failed:
						return errors.New("index build failed")
					case common.IndexState_Finished:
						return nil
					default:
					}
				}
			}
			time.Sleep(100 * time.Millisecond) // wait 100ms
		}
	}
	return nil
}

type DescribeIndexRequestOption func(req *server.DescribeIndexRequest)

func SetCollectionNameForDescribeIndex(name string) DescribeIndexRequestOption {
	return func(req *server.DescribeIndexRequest) {
		req.CollectionName = name
	}
}

func SetIndexNameForDescribeIndex(name string) DescribeIndexRequestOption {
	return func(req *server.DescribeIndexRequest) {
		req.IndexName = name
	}
}

func SetFieldNameForDescribeIndex(name string) DescribeIndexRequestOption {
	return func(req *server.DescribeIndexRequest) {
		req.FieldName = name
	}
}

// DescribeIndex describe index
func (c *grpcClient) DescribeIndex(ctx context.Context, opts ...DescribeIndexRequestOption) ([]*server.IndexDescription, error) {
	if c.Service == nil {
		return []*server.IndexDescription{}, client.ErrClientNotReady
	}

	req := &server.DescribeIndexRequest{}
	for _, opt := range opts {
		opt(req)
	}
	if req.GetCollectionName() == "" {
		return []*server.IndexDescription{}, errors.New("collection name can not be empty")
	}
	resp, err := c.Service.DescribeIndex(ctx, req)
	if err != nil {
		return []*server.IndexDescription{}, err
	}
	if err := handleRespStatus(resp.GetStatus()); err != nil {
		return []*server.IndexDescription{}, err
	}
	return resp.GetIndexDescriptions(), nil
}

type DropIndexOption func(request *server.DropIndexRequest)

func SetCollectionNameForDropIndex(name string) DropIndexOption {
	return func(req *server.DropIndexRequest) {
		req.CollectionName = name
	}
}
func SetIndexNameForDropIndex(name string) DropIndexOption {
	return func(req *server.DropIndexRequest) {
		req.IndexName = name
	}
}

func SetFieldNameForDroIndex(name string) DropIndexOption {
	return func(req *server.DropIndexRequest) {
		req.FieldName = name
	}
}

// DropIndex drop index from collection
func (c *grpcClient) DropIndex(ctx context.Context, opts ...DropIndexOption) error {
	if c.Service == nil {
		return client.ErrClientNotReady
	}
	req := &server.DropIndexRequest{}
	for _, opt := range opts {
		opt(req)
	}
	if req.GetCollectionName() == "" {
		return errors.New("collection name can not be empty")
	}

	resp, err := c.Service.DropIndex(ctx, req)
	if err != nil {
		return err
	}
	return handleRespStatus(resp)
}
