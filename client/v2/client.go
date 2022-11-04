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
	"fmt"
	"math"
	"time"

	grpc_middleware "github.com/grpc-ecosystem/go-grpc-middleware"
	grpc_retry "github.com/grpc-ecosystem/go-grpc-middleware/retry"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"

	server "github.com/milvus-io/milvus-proto/go-api/milvuspb"
	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
	"google.golang.org/grpc"
)

// Client is the interface used to communicate with Milvus
type Client interface {
	// Close the remaining connection resources
	Close() error

	// -- collection --

	// ListCollections list collections from connection
	ListCollections(ctx context.Context) ([]*entity.Collection, error)
	// CreateCollection create collection using provided schema
	CreateCollection(ctx context.Context, schema *entity.Schema, shardsNum int32, opts ...client.CreateCollectionOption) error
	// DescribeCollection describe collection meta
	DescribeCollection(ctx context.Context, collName string) (*entity.Collection, error)
	// DropCollection drop the specified collection
	DropCollection(ctx context.Context, collName string) error
	// GetCollectionStatistics get collection statistics
	GetCollectionStatistics(ctx context.Context, collName string) (map[string]string, error)
	// LoadCollection load collection into memory
	LoadCollection(ctx context.Context, collName string, async bool, opts ...client.LoadCollectionOption) error
	// ReleaseCollection release loaded collection
	ReleaseCollection(ctx context.Context, collName string) error
	// HasCollection check whether collection exists
	HasCollection(ctx context.Context, collName string) (bool, error)

	// CreateAlias creates an alias for collection
	CreateAlias(ctx context.Context, collName string, alias string) error
	// DropAlias drops the specified Alias
	DropAlias(ctx context.Context, alias string) error
	// AlterAlias changes collection alias to provided alias
	AlterAlias(ctx context.Context, collName string, alias string) error

	// GetReplicas gets the replica groups as well as their querynodes and shards information
	GetReplicas(ctx context.Context, collName string) ([]*entity.ReplicaGroup, error)

	// -- authentication --

	// CreateCredential create new user and password
	CreateCredential(ctx context.Context, username string, password string) error
	// UpdateCredential update password for a user
	UpdateCredential(ctx context.Context, username string, oldPassword string, newPassword string) error
	// DeleteCredential delete a user
	DeleteCredential(ctx context.Context, username string) error
	// ListCredUsers list all usernames
	ListCredUsers(ctx context.Context) ([]string, error)

	// -- partition --

	// CreatePartition create partition for collection
	CreatePartition(ctx context.Context, collName string, partitionName string) error
	// DropPartition drop partition from collection
	DropPartition(ctx context.Context, collName string, partitionName string) error
	// ShowPartitions list all partitions from collection
	ShowPartitions(ctx context.Context, collName string) ([]*entity.Partition, error)
	// HasPartition check whether partition exists in collection
	HasPartition(ctx context.Context, collName string, partitionName string) (bool, error)
	// LoadPartitions load partitions into memory
	LoadPartitions(ctx context.Context, collName string, partitionNames []string, async bool) error
	// ReleasePartitions release partitions
	ReleasePartitions(ctx context.Context, collName string, partitionNames []string) error

	// -- segment --
	GetPersistentSegmentInfo(ctx context.Context, collName string) ([]*entity.Segment, error)
	// -- index --

	// CreateIndex create index for field of specified collection
	// currently index naming is not supported, so only one index on vector field is supported
	CreateIndex(ctx context.Context, async bool, idx entity.Index, opts ...CreateIndexRequestOption) error
	// DescribeIndex describe index on collection
	// currently index naming is not supported, so only one index on vector field is supported
	DescribeIndex(ctx context.Context, opts ...DescribeIndexRequestOption) ([]*server.IndexDescription, error)
	// DropIndex drop index from collection with specified field name
	DropIndex(ctx context.Context, opts ...DropIndexOption) error
	// GetIndexState get index state with specified collection and field name
	// index naming is not supported yet
	// Deprecated use DescribeIndexV2 instead.
	GetIndexState(ctx context.Context, collName string, fieldName string) (entity.IndexState, error)

	// -- basic operation --

	// Insert column-based data into collection, returns id column values
	Insert(ctx context.Context, collName string, partitionName string, columns ...entity.Column) (entity.Column, error)
	// Flush flush collection, specified
	Flush(ctx context.Context, collName string, async bool) error
	// DeleteByPks deletes entries related to provided primary keys
	DeleteByPks(ctx context.Context, collName string, partitionName string, ids entity.Column) error
	// Search search with bool expression
	Search(ctx context.Context, collName string, partitions []string,
		expr string, outputFields []string, vectors []entity.Vector, vectorField string, metricType entity.MetricType, topK int, sp entity.SearchParam, opts ...client.SearchQueryOptionFunc) ([]client.SearchResult, error)
	// QueryByPks query record by specified primary key(s)
	QueryByPks(ctx context.Context, collectionName string, partitionNames []string, ids entity.Column, outputFields []string, opts ...client.SearchQueryOptionFunc) ([]entity.Column, error)

	// CalcDistance calculate the distance between vectors specified by ids or provided
	CalcDistance(ctx context.Context, collName string, partitions []string,
		metricType entity.MetricType, opLeft, opRight entity.Column) (entity.Column, error)

	// -- row based apis --

	// CreateCollectionByRow create collection by row
	CreateCollectionByRow(ctx context.Context, row entity.Row, shardNum int32) error
	// InsertByRows insert by rows
	InsertByRows(ctx context.Context, collName string, paritionName string, rows []entity.Row) (entity.Column, error)

	// ManualCompaction triggers a compaction on provided collection
	ManualCompaction(ctx context.Context, collName string, toleranceDuration time.Duration) (int64, error)
	// GetCompactionState get compaction state of provided compaction id
	GetCompactionState(ctx context.Context, id int64) (entity.CompactionState, error)
	// GetCompactionStateWithPlans get compaction state with plans of provided compaction id
	GetCompactionStateWithPlans(ctx context.Context, id int64) (entity.CompactionState, []entity.CompactionPlan, error)

	// BulkInsert import data files(json, numpy, etc.) on MinIO/S3 storage, read and parse them into sealed segments
	BulkInsert(ctx context.Context, collName string, partitionName string, rowBased bool, files []string, opts ...client.BulkInsertOption) ([]int64, error)
	// GetBulkInsertState checks import task state
	GetBulkInsertState(ctx context.Context, taskID int64) (*entity.BulkInsertTaskState, error)
	// ListBulkInsertTasks list state of all import tasks
	ListBulkInsertTasks(ctx context.Context, collName string, limit int64) ([]*entity.BulkInsertTaskState, error)
}

// grpcClient, uses default grpc service definition to connect with Milvus2.0
type grpcClient struct {
	client.GrpcClient
	//conn    *grpc.ClientConn           // grpc connection instance
	//service server.MilvusServiceClient // service client stub
}

// connect connect to service
func (c *grpcClient) connect(ctx context.Context, addr string, opts ...grpc.DialOption) error {
	if addr == "" {
		return fmt.Errorf("address is empty")
	}
	conn, err := grpc.Dial(addr, opts...)
	if err != nil {
		return err
	}

	c.Conn = conn
	c.Service = server.NewMilvusServiceClient(c.Conn)
	return nil
}

// Close the connection
func (c *grpcClient) Close() error {
	if c.Conn != nil {
		err := c.Conn.Close()
		c.Conn = nil
		return err
	}
	return nil
}

// NewGrpcClient create client with grpc addr
// the `Connect` API will be called for you
// dialOptions contains the dial option(s) that control the grpc dialing process
func NewGrpcClient(ctx context.Context, addr string, dialOptions ...grpc.DialOption) (Client, error) {
	c := &grpcClient{}
	if len(dialOptions) == 0 {
		return NewDefaultGrpcClient(ctx, addr)
	}

	if err := c.connect(ctx, addr, dialOptions...); err != nil {
		return nil, err
	}

	return c, nil
}

func NewDefaultGrpcClient(ctx context.Context, addr string) (Client, error) {
	c := &grpcClient{}
	defaultOpts := append(client.DefaultGrpcOpts,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithUnaryInterceptor(
			grpc_middleware.ChainUnaryClient(
				grpc_retry.UnaryClientInterceptor(
					grpc_retry.WithMax(6),
					grpc_retry.WithBackoff(func(attempt uint) time.Duration {
						return 60 * time.Millisecond * time.Duration(math.Pow(3, float64(attempt)))
					}),
					grpc_retry.WithCodes(codes.Unavailable, codes.ResourceExhausted)),
				client.RetryOnRateLimitInterceptor(10, func(ctx context.Context, attempt uint) time.Duration {
					return 10 * time.Millisecond * time.Duration(math.Pow(3, float64(attempt)))
				}),
			),
		),
	)
	err := c.connect(ctx, addr, defaultOpts...)
	if err != nil {
		return nil, err
	}
	return c, nil
}
