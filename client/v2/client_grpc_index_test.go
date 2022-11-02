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
	"log"
	"math/rand"
	"net"
	"testing"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/test/bufconn"

	"github.com/golang/protobuf/proto"
	common "github.com/milvus-io/milvus-proto/go-api/commonpb"
	server "github.com/milvus-io/milvus-proto/go-api/milvuspb"
	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
	"github.com/stretchr/testify/assert"
)

const (
	bufSzie = 1024 * 1024
)

var (
	lis  *bufconn.Listener
	mock *client.MockServer
)

// TestMain establishes mock grpc server to testing client behavior
func TestMain(m *testing.M) {
	rand.Seed(time.Now().Unix())
	lis = bufconn.Listen(bufSzie)
	s := grpc.NewServer()
	mock = &client.MockServer{
		Injections: make(map[client.ServiceMethod]client.TestInjection),
	}
	server.RegisterMilvusServiceServer(s, mock)
	go func() {
		if err := s.Serve(lis); err != nil {
			log.Fatalf("Server exited with error: %v", err)
		}
	}()
	m.Run()
	//	lis.Close()
}

// use bufconn dialer
func bufDialer(context.Context, string) (net.Conn, error) {
	return lis.Dial()
}

func testClient(ctx context.Context, t *testing.T) Client {
	c, err := NewGrpcClient(ctx, "bufnet", grpc.WithBlock(),
		grpc.WithInsecure(), grpc.WithContextDialer(bufDialer))

	if !assert.Nil(t, err) || !assert.NotNil(t, c) {
		t.FailNow()
	}
	return c
}

func TestGrpcClient_CreateIndex(t *testing.T) {
	ctx := context.Background()
	c := testClient(ctx, t)

	const (
		collectionName   = "coll_index"
		defaultIndexName = "_default_idx_101"
		indexName        = "indexName"
		emptyIndexName   = ""
		fieldName        = "fieldName"
	)

	var (
		flag  = false
		count = 0
	)

	idx, err := entity.NewIndexHNSW(entity.L2, 16, 64)
	assert.Nil(t, err)
	if !assert.NotNil(t, idx) {
		t.FailNow()
	}
	mock.SetInjection(client.MCreateIndex, func(_ context.Context, raw proto.Message) (proto.Message, error) {
		req, ok := raw.(*server.CreateIndexRequest)
		if !ok {
			return client.BadRequestStatus()
		}
		assert.Equal(t, collectionName, req.GetCollectionName())
		assert.Equal(t, fieldName, req.GetFieldName())
		//assert.Equal(t, defaultIndexName, req.GetIndexName())
		return client.SuccessStatus()
	})

	mock.SetInjection(client.MDescribeIndex, func(_ context.Context, message proto.Message) (proto.Message, error) {
		var err2 error
		req, ok := message.(*server.DescribeIndexRequest)
		resp := &server.DescribeIndexResponse{}
		if !ok {
			resp.Status, err2 = client.BadRequestStatus()
			return resp.Status, err2
		}
		assert.Equal(t, collectionName, req.GetCollectionName())
		assert.Equal(t, fieldName, req.GetFieldName())
		//assert.Equal(t, defaultIndexName, req.GetIndexName())

		count++

		if req.GetIndexName() == emptyIndexName {
			resp.IndexDescriptions = []*server.IndexDescription{
				{
					IndexName:            defaultIndexName,
					IndexID:              1,
					Params:               entity.MapKvPairs(idx.Params()),
					FieldName:            req.GetFieldName(),
					IndexedRows:          9000,
					TotalRows:            10000,
					State:                common.IndexState_InProgress,
					IndexStateFailReason: "",
				},
				{
					IndexName:            indexName,
					IndexID:              2,
					Params:               entity.MapKvPairs(idx.Params()),
					FieldName:            req.GetFieldName(),
					IndexedRows:          2500,
					TotalRows:            3000,
					State:                common.IndexState_InProgress,
					IndexStateFailReason: "",
				},
			}
		} else {
			resp.IndexDescriptions = []*server.IndexDescription{
				{
					IndexName:            req.GetIndexName(),
					IndexID:              0,
					Params:               entity.MapKvPairs(idx.Params()),
					FieldName:            req.GetFieldName(),
					IndexedRows:          9000,
					TotalRows:            10000,
					State:                common.IndexState_InProgress,
					IndexStateFailReason: "",
				},
			}
		}

		if count >= 10 {
			for i := range resp.GetIndexDescriptions() {
				resp.IndexDescriptions[i].State = common.IndexState_Finished
			}
			flag = true
		}

		resp.Status, err2 = client.SuccessStatus()
		return resp, err2
	})

	t.Run("normal", func(t *testing.T) {
		err = c.CreateIndex(ctx, true, idx, SetCollectionNameForCreateIndex(collectionName),
			SetFieldNameForCreateIndex(fieldName), SetIndexNameForCreateIndex(defaultIndexName))
		assert.NoError(t, err)
	})

	t.Run("sync mode", func(t *testing.T) {
		err = c.CreateIndex(ctx, false, idx, SetCollectionNameForCreateIndex(collectionName),
			SetFieldNameForCreateIndex(fieldName), SetIndexNameForCreateIndex(defaultIndexName))
		assert.NoError(t, err)
		assert.True(t, flag)
	})

	t.Run("empty collection name", func(t *testing.T) {
		err = c.CreateIndex(ctx, false, idx,
			SetFieldNameForCreateIndex(fieldName), SetIndexNameForCreateIndex(defaultIndexName))
		assert.Error(t, err)
	})

	t.Run("empty field name", func(t *testing.T) {
		err = c.CreateIndex(ctx, false, idx, SetCollectionNameForCreateIndex(collectionName),
			SetIndexNameForCreateIndex(defaultIndexName))
		assert.Error(t, err)
	})

	t.Run("empty index name", func(t *testing.T) {
		count = 0
		flag = false
		err = c.CreateIndex(ctx, false, idx, SetCollectionNameForCreateIndex(collectionName),
			SetFieldNameForCreateIndex(fieldName))
		assert.NoError(t, err)
		assert.True(t, flag)
	})

	t.Run("index failed", func(t *testing.T) {
		mock.SetInjection(client.MDescribeIndex, func(ctx context.Context, message proto.Message) (proto.Message, error) {
			var err2 error
			req, ok := message.(*server.DescribeIndexRequest)
			resp := &server.DescribeIndexResponse{}
			if !ok {
				resp.Status, err2 = client.BadRequestStatus()
				return resp, err2
			}
			resp.IndexDescriptions = []*server.IndexDescription{
				{
					IndexName:            req.GetIndexName(),
					IndexID:              1,
					Params:               nil,
					FieldName:            req.GetFieldName(),
					IndexedRows:          1000,
					TotalRows:            0,
					State:                common.IndexState_Failed,
					IndexStateFailReason: "",
				},
			}
			return resp, err2
		})

		err = c.CreateIndex(ctx, false, idx, SetCollectionNameForCreateIndex(collectionName),
			SetFieldNameForCreateIndex(fieldName), SetIndexNameForCreateIndex(defaultIndexName))
		assert.Error(t, err)
	})

	t.Run("bad status", func(t *testing.T) {
		mock.SetInjection(client.MDescribeIndex, func(ctx context.Context, message proto.Message) (proto.Message, error) {
			var err2 error
			resp := &server.DescribeIndexResponse{}
			resp.Status, err2 = client.BadStatus()
			return resp, err2
		})
		err = c.CreateIndex(ctx, false, idx, SetCollectionNameForCreateIndex(collectionName),
			SetFieldNameForCreateIndex(fieldName), SetIndexNameForCreateIndex(defaultIndexName))
		assert.Error(t, err)

		mock.SetInjection(client.MCreateIndex, func(ctx context.Context, message proto.Message) (proto.Message, error) {
			return client.BadStatus()
		})

		err = c.CreateIndex(ctx, false, idx, SetCollectionNameForCreateIndex(collectionName),
			SetFieldNameForCreateIndex(fieldName), SetIndexNameForCreateIndex(defaultIndexName))
		assert.Error(t, err)
	})

	t.Run("error", func(t *testing.T) {
		mock.SetInjection(client.MDescribeIndex, func(ctx context.Context, message proto.Message) (proto.Message, error) {
			var err2 error
			resp := &server.DescribeIndexResponse{}
			resp.Status, err2 = client.BadRequestStatus()
			return resp, err2
		})
		mock.SetInjection(client.MCreateIndex, func(ctx context.Context, message proto.Message) (proto.Message, error) {
			return client.SuccessStatus()
		})

		err = c.CreateIndex(ctx, false, idx, SetCollectionNameForCreateIndex(collectionName),
			SetFieldNameForCreateIndex(fieldName), SetIndexNameForCreateIndex(defaultIndexName))
		assert.Error(t, err)

		mock.SetInjection(client.MCreateIndex, func(ctx context.Context, message proto.Message) (proto.Message, error) {
			return client.BadRequestStatus()
		})

		err = c.CreateIndex(ctx, false, idx, SetCollectionNameForCreateIndex(collectionName),
			SetFieldNameForCreateIndex(fieldName), SetIndexNameForCreateIndex(defaultIndexName))
		assert.Error(t, err)
	})
}

func TestGrpcClient_DescribeIndex(t *testing.T) {
	ctx := context.Background()
	c := testClient(ctx, t)

	const (
		collectionName   = "coll_index"
		defaultIndexName = "_default_idx_101"
		indexName        = "indexName"
		emptyIndexName   = ""
		fieldName        = "fieldName"
	)

	idx, err := entity.NewIndexHNSW(entity.L2, 16, 64)
	assert.Nil(t, err)
	if !assert.NotNil(t, idx) {
		t.FailNow()
	}

	t.Run("normal", func(t *testing.T) {
		mock.SetInjection(client.MDescribeIndex, func(ctx context.Context, message proto.Message) (proto.Message, error) {
			var err2 error
			req, ok := message.(*server.DescribeIndexRequest)
			resp := &server.DescribeIndexResponse{}
			if !ok {
				resp.Status, err2 = client.BadRequestStatus()
				return resp, err2
			}
			assert.Equal(t, collectionName, req.GetCollectionName())

			if req.GetIndexName() == emptyIndexName {
				resp.IndexDescriptions = []*server.IndexDescription{
					{
						IndexName:            defaultIndexName,
						IndexID:              1,
						Params:               entity.MapKvPairs(idx.Params()),
						FieldName:            req.GetFieldName(),
						IndexedRows:          9000,
						TotalRows:            10000,
						State:                common.IndexState_InProgress,
						IndexStateFailReason: "",
					},
					{
						IndexName:            indexName,
						IndexID:              2,
						Params:               entity.MapKvPairs(idx.Params()),
						FieldName:            req.GetFieldName(),
						IndexedRows:          2500,
						TotalRows:            3000,
						State:                common.IndexState_InProgress,
						IndexStateFailReason: "",
					},
				}
			} else {
				resp.IndexDescriptions = []*server.IndexDescription{
					{
						IndexName:            req.GetIndexName(),
						IndexID:              1,
						Params:               entity.MapKvPairs(idx.Params()),
						FieldName:            req.GetFieldName(),
						IndexedRows:          9000,
						TotalRows:            10000,
						State:                common.IndexState_InProgress,
						IndexStateFailReason: "",
					},
				}
			}
			resp.Status, err2 = client.SuccessStatus()
			return resp, err2
		})

		indexes, err := c.DescribeIndex(ctx, SetCollectionNameForDescribeIndex(collectionName),
			SetIndexNameForDescribeIndex(indexName))
		assert.NoError(t, err)
		assert.Equal(t, 1, len(indexes))
		assert.Equal(t, indexName, indexes[0].GetIndexName())
	})

	t.Run("empty collection name", func(t *testing.T) {
		indexes, err := c.DescribeIndex(ctx,
			SetIndexNameForDescribeIndex(indexName))
		assert.Error(t, err)
		assert.Equal(t, 0, len(indexes))
	})

	t.Run("empty index name", func(t *testing.T) {
		indexes, err := c.DescribeIndex(ctx, SetCollectionNameForDescribeIndex(collectionName))
		assert.NoError(t, err)
		assert.Equal(t, 2, len(indexes))
	})

	t.Run("bad status", func(t *testing.T) {
		mock.SetInjection(client.MDescribeIndex, func(ctx context.Context, message proto.Message) (proto.Message, error) {
			return &server.DescribeIndexResponse{
				Status: &common.Status{
					ErrorCode: common.ErrorCode_IndexNotExist,
					Reason:    "index not exist",
				},
			}, nil
		})
		indexes, err := c.DescribeIndex(ctx, SetCollectionNameForDescribeIndex(collectionName))
		assert.Error(t, err)
		assert.Equal(t, 0, len(indexes))
	})

	t.Run("error", func(t *testing.T) {
		mock.SetInjection(client.MDescribeIndex, func(ctx context.Context, message proto.Message) (proto.Message, error) {
			return &server.DescribeIndexResponse{
				Status: &common.Status{
					ErrorCode: common.ErrorCode_UnexpectedError,
					Reason:    "",
				},
			}, errors.New("error")
		})
		indexes, err := c.DescribeIndex(ctx, SetCollectionNameForDescribeIndex(collectionName))
		assert.Error(t, err)
		assert.Equal(t, 0, len(indexes))
	})
}

func TestGrpcClient_DropIndex(t *testing.T) {
	ctx := context.Background()
	c := testClient(ctx, t)

	const (
		collectionName   = "coll_index"
		defaultIndexName = "_default_idx_101"
		indexName        = "indexName"
		emptyIndexName   = ""
		fieldName        = "fieldName"
	)

	idx, err := entity.NewIndexHNSW(entity.L2, 16, 64)
	assert.Nil(t, err)
	if !assert.NotNil(t, idx) {
		t.FailNow()
	}

	mock.SetInjection(client.MDropIndex, func(ctx context.Context, message proto.Message) (proto.Message, error) {
		req, ok := message.(*server.DropIndexRequest)
		if !ok {
			return client.BadRequestStatus()
		}
		assert.Equal(t, collectionName, req.GetCollectionName())

		return client.SuccessStatus()
	})

	t.Run("normal", func(t *testing.T) {
		err = c.DropIndex(ctx, SetCollectionNameForDropIndex(collectionName), SetFieldNameForDroIndex(fieldName),
			SetIndexNameForDropIndex(indexName))
		assert.NoError(t, err)
	})

	t.Run("empty collection name", func(t *testing.T) {
		err = c.DropIndex(ctx, SetFieldNameForDroIndex(fieldName),
			SetIndexNameForDropIndex(indexName))
		assert.Error(t, err)
	})

	t.Run("empty field name", func(t *testing.T) {
		err = c.DropIndex(ctx, SetCollectionNameForDropIndex(collectionName), SetIndexNameForDropIndex(indexName))
		assert.NoError(t, err)
	})

	t.Run("empty index name", func(t *testing.T) {
		err = c.DropIndex(ctx, SetCollectionNameForDropIndex(collectionName), SetFieldNameForDroIndex(fieldName))
		assert.NoError(t, err)

		err = c.DropIndex(ctx, SetCollectionNameForDropIndex(collectionName))
		assert.NoError(t, err)
	})

	t.Run("bad status", func(t *testing.T) {
		mock.SetInjection(client.MDropIndex, func(ctx context.Context, message proto.Message) (proto.Message, error) {
			return client.BadStatus()
		})
		err = c.DropIndex(ctx, SetCollectionNameForDropIndex(collectionName))
		assert.Error(t, err)
	})

	t.Run("error", func(t *testing.T) {
		mock.SetInjection(client.MDropIndex, func(ctx context.Context, message proto.Message) (proto.Message, error) {
			return &common.Status{
				ErrorCode: common.ErrorCode_UnexpectedError,
				Reason:    "fail",
			}, errors.New("error")
		})
		err = c.DropIndex(ctx, SetCollectionNameForDropIndex(collectionName))
		assert.Error(t, err)
	})
}
