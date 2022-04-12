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

// Package client provides milvus client functions
package client

import (
	"context"
	"time"

	"github.com/milvus-io/milvus-sdk-go/v2/entity"
	"google.golang.org/grpc"
)

// Client is the interface used to communicate with Milvus
// The common usage is like follow
//		c, err := client.NewGrpcClient(context.Background, "address-to-milvus") // or other creation func maybe added later
//		if err != nil {
//		//handle err
//		}
//		// start doing things with client instance, note that there is no need to call Connect since NewXXXClient will do that for you
type Client interface {
	// Close close the remaining connection resources
	Close() error

	// -- collection --

	// ListCollections list collections from connection
	ListCollections(ctx context.Context) ([]*entity.Collection, error)
	// CreateCollection create collection using provided schema
	CreateCollection(ctx context.Context, schema *entity.Schema, shardsNum int32) error
	// DescribeCollection describe collection meta
	DescribeCollection(ctx context.Context, collName string) (*entity.Collection, error)
	// DropCollection drop the specified collection
	DropCollection(ctx context.Context, collName string) error
	// GetCollectionStatistics get collection statistics
	GetCollectionStatistics(ctx context.Context, collName string) (map[string]string, error)
	// LoadCollection load collection into memory
	LoadCollection(ctx context.Context, collName string, async bool) error
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

	// -- index --

	// CreateIndex create index for field of specified collection
	// currently index naming is not supported, so only one index on vector field is supported
	CreateIndex(ctx context.Context, collName string, fieldName string, idx entity.Index, async bool) error
	// DescribeIndex describe index on collection
	// currently index naming is not supported, so only one index on vector field is supported
	DescribeIndex(ctx context.Context, collName string, fieldName string) ([]entity.Index, error)
	// DropIndex drop index from collection with specified field name
	DropIndex(ctx context.Context, collName string, fieldName string) error
	// GetIndexState get index state with specified collection and field name
	// index naming is not supported yet
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
		expr string, outputFields []string, vectors []entity.Vector, vectorField string, metricType entity.MetricType, topK int, sp entity.SearchParam) ([]SearchResult, error)
	// QueryByPks query record by specified primary key(s)
	QueryByPks(ctx context.Context, collectionName string, partitionNames []string, ids entity.Column, outputFields []string) ([]entity.Column, error)

	// CalcDistance calculate the distance between vectors specified by ids or provided
	CalcDistance(ctx context.Context, collName string, partitions []string,
		metricType entity.MetricType, opLeft, opRight entity.Column) (entity.Column, error)

	// -- row basd apis --
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
}

// SearchResult contains the result from Search api of client
// IDs is the auto generated id values for the entities
// Fields contains the data of `outputFieleds` specified or all columns if non
// Scores is actually the distance between the vector current record contains and the search target vector
type SearchResult struct {
	ResultCount int             // the returning entry count
	IDs         entity.Column   // auto generated id, can be mapped to the columns from `Insert` API
	Fields      []entity.Column // output field data
	Scores      []float32       // distance to the target vector
	Err         error           // search error if any
}

// alias type for context field key
type grpcKey int

const (
	dialOption grpcKey = 1
)

// NewGrpcClient create client with grpc addr
// the `Connect` API will be called for you
// dialOptions contains the dial option(s) that control the grpc dialing process
func NewGrpcClient(ctx context.Context, addr string, dialOptions ...grpc.DialOption) (Client, error) {
	c := &grpcClient{}
	err := c.connect(ctx, addr, dialOptions...)
	if err != nil {
		return nil, err
	}
	return c, nil
}
