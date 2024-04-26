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

	"google.golang.org/grpc"

	"github.com/milvus-io/milvus-proto/go-api/v2/msgpb"

	"github.com/milvus-io/milvus-sdk-go/v2/entity"
)

// Client is the interface used to communicate with Milvus
type Client interface {
	// -- client --
	// Close close the remaining connection resources
	Close() error

	// UsingDatabase for database operation after this function call.
	// All request in any goroutine will be applied to new database on the same client. e.g.
	// 1. goroutine A access DB1.
	// 2. goroutine B call UsingDatabase(ctx, "DB2").
	// 3. goroutine A access DB2 after 2.
	UsingDatabase(ctx context.Context, dbName string) error

	// -- database --
	// ListDatabases list all database in milvus cluster.
	ListDatabases(ctx context.Context) ([]entity.Database, error)
	// CreateDatabase create database with the given name.
	CreateDatabase(ctx context.Context, dbName string, opts ...CreateDatabaseOption) error
	// DropDatabase drop database with the given db name.
	DropDatabase(ctx context.Context, dbName string, opts ...DropDatabaseOption) error

	// -- collection --

	// NewCollection intializeds a new collection with pre defined attributes
	NewCollection(ctx context.Context, collName string, dimension int64, opts ...CreateCollectionOption) error
	// ListCollections list collections from connection
	ListCollections(ctx context.Context) ([]*entity.Collection, error)
	// CreateCollection create collection using provided schema
	CreateCollection(ctx context.Context, schema *entity.Schema, shardsNum int32, opts ...CreateCollectionOption) error
	// DescribeCollection describe collection meta
	DescribeCollection(ctx context.Context, collName string) (*entity.Collection, error)
	// DropCollection drop the specified collection
	DropCollection(ctx context.Context, collName string, opts ...DropCollectionOption) error
	// GetCollectionStatistics get collection statistics
	GetCollectionStatistics(ctx context.Context, collName string) (map[string]string, error)
	// LoadCollection load collection into memory
	LoadCollection(ctx context.Context, collName string, async bool, opts ...LoadCollectionOption) error
	// ReleaseCollection release loaded collection
	ReleaseCollection(ctx context.Context, collName string, opts ...ReleaseCollectionOption) error
	// HasCollection check whether collection exists
	HasCollection(ctx context.Context, collName string) (bool, error)
	// RenameCollection performs renaming for provided collection.
	RenameCollection(ctx context.Context, collName, newName string) error
	// AlterCollection changes collection attributes.
	AlterCollection(ctx context.Context, collName string, attrs ...entity.CollectionAttribute) error

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
	CreatePartition(ctx context.Context, collName string, partitionName string, opts ...CreatePartitionOption) error
	// DropPartition drop partition from collection
	DropPartition(ctx context.Context, collName string, partitionName string, opts ...DropPartitionOption) error
	// ShowPartitions list all partitions from collection
	ShowPartitions(ctx context.Context, collName string) ([]*entity.Partition, error)
	// HasPartition check whether partition exists in collection
	HasPartition(ctx context.Context, collName string, partitionName string) (bool, error)
	// LoadPartitions load partitions into memory
	LoadPartitions(ctx context.Context, collName string, partitionNames []string, async bool, opts ...LoadPartitionsOption) error
	// ReleasePartitions release partitions
	ReleasePartitions(ctx context.Context, collName string, partitionNames []string, opts ...ReleasePartitionsOption) error

	// -- segment --
	GetPersistentSegmentInfo(ctx context.Context, collName string) ([]*entity.Segment, error)
	// -- index --

	// CreateIndex create index for field of specified collection
	// currently index naming is not supported, so only one index on vector field is supported
	CreateIndex(ctx context.Context, collName string, fieldName string, idx entity.Index, async bool, opts ...IndexOption) error
	// DescribeIndex describe index on collection
	// currently index naming is not supported, so only one index on vector field is supported
	DescribeIndex(ctx context.Context, collName string, fieldName string, opts ...IndexOption) ([]entity.Index, error)
	// DropIndex drop index from collection with specified field name
	DropIndex(ctx context.Context, collName string, fieldName string, opts ...IndexOption) error
	// GetIndexState get index state with specified collection and field name
	// index naming is not supported yet
	GetIndexState(ctx context.Context, collName string, fieldName string, opts ...IndexOption) (entity.IndexState, error)
	// AlterIndex modifies the index params.
	AlterIndex(ctx context.Context, collName, indexName string, opts ...IndexOption) error

	// -- basic operation --

	// Insert column-based data into collection, returns id column values
	Insert(ctx context.Context, collName string, partitionName string, columns ...entity.Column) (entity.Column, error)
	// Flush collection, specified
	Flush(ctx context.Context, collName string, async bool, opts ...FlushOption) error
	// FlushV2 flush collection, specified, return newly sealed segmentIds, all flushed segmentIds of the collection, seal time and error
	// currently it is only used in milvus-backup(https://github.com/zilliztech/milvus-backup)
	FlushV2(ctx context.Context, collName string, async bool, opts ...FlushOption) ([]int64, []int64, int64, error)
	// DeleteByPks deletes entries related to provided primary keys
	DeleteByPks(ctx context.Context, collName string, partitionName string, ids entity.Column) error
	// Delete deletes entries match expression
	Delete(ctx context.Context, collName string, partitionName string, expr string) error
	// Upsert column-based data of collection, returns id column values
	Upsert(ctx context.Context, collName string, partitionName string, columns ...entity.Column) (entity.Column, error)
	// Search with bool expression
	Search(ctx context.Context, collName string, partitions []string,
		expr string, outputFields []string, vectors []entity.Vector, vectorField string, metricType entity.MetricType, topK int, sp entity.SearchParam, opts ...SearchQueryOptionFunc) ([]SearchResult, error)
	// QueryByPks query record by specified primary key(s).
	QueryByPks(ctx context.Context, collectionName string, partitionNames []string, ids entity.Column, outputFields []string, opts ...SearchQueryOptionFunc) (ResultSet, error)
	// Query performs query records with boolean expression.
	Query(ctx context.Context, collectionName string, partitionNames []string, expr string, outputFields []string, opts ...SearchQueryOptionFunc) (ResultSet, error)
	// Get grabs the inserted entities using the primary key from the Collection.
	Get(ctx context.Context, collectionName string, ids entity.Column, opts ...GetOption) (ResultSet, error)

	// CalcDistance calculate the distance between vectors specified by ids or provided
	CalcDistance(ctx context.Context, collName string, partitions []string,
		metricType entity.MetricType, opLeft, opRight entity.Column) (entity.Column, error)

	// -- row based apis --

	// CreateCollectionByRow create collection by row
	CreateCollectionByRow(ctx context.Context, row entity.Row, shardNum int32) error
	// DEPRECATED
	// InsertByRows insert by rows
	InsertByRows(ctx context.Context, collName string, paritionName string, rows []entity.Row) (entity.Column, error)
	// InsertRows insert with row base data.
	InsertRows(ctx context.Context, collName string, partitionName string, rows []interface{}) (entity.Column, error)

	// ManualCompaction triggers a compaction on provided collection
	ManualCompaction(ctx context.Context, collName string, toleranceDuration time.Duration) (int64, error)
	// GetCompactionState get compaction state of provided compaction id
	GetCompactionState(ctx context.Context, id int64) (entity.CompactionState, error)
	// GetCompactionStateWithPlans get compaction state with plans of provided compaction id
	GetCompactionStateWithPlans(ctx context.Context, id int64) (entity.CompactionState, []entity.CompactionPlan, error)

	// BulkInsert import data files(json, numpy, etc.) on MinIO/S3 storage, read and parse them into sealed segments
	BulkInsert(ctx context.Context, collName string, partitionName string, files []string, opts ...BulkInsertOption) (int64, error)
	// GetBulkInsertState checks import task state
	GetBulkInsertState(ctx context.Context, taskID int64) (*entity.BulkInsertTaskState, error)
	// ListBulkInsertTasks list state of all import tasks
	ListBulkInsertTasks(ctx context.Context, collName string, limit int64) ([]*entity.BulkInsertTaskState, error)

	// CreateRole creates a role entity in Milvus.
	CreateRole(ctx context.Context, name string) error
	// DropRole drops a role entity in Milvus.
	DropRole(ctx context.Context, name string) error
	// AddUserRole adds one role for user.
	AddUserRole(ctx context.Context, username string, role string) error
	// RemoveUserRole removes one role from user.
	RemoveUserRole(ctx context.Context, username string, role string) error
	// ListRoles lists the role objects in system.
	ListRoles(ctx context.Context) ([]entity.Role, error)
	// ListUsers lists the user objects in system.
	ListUsers(ctx context.Context) ([]entity.User, error)
	// DescribeUser describes specific user attributes in the system
	DescribeUser(ctx context.Context, username string) (entity.UserDescription, error)
	// DescribeUsers describe all users attributes in the system
	DescribeUsers(ctx context.Context) ([]entity.UserDescription, error)
	// Grant adds privilege for role.
	Grant(ctx context.Context, role string, objectType entity.PriviledgeObjectType, object string) error
	// Revoke removes privilege from role.
	Revoke(ctx context.Context, role string, objectType entity.PriviledgeObjectType, object string) error

	// GetLoadingProgress get the collection or partitions loading progress
	GetLoadingProgress(ctx context.Context, collectionName string, partitionNames []string) (int64, error)
	// GetLoadState get the collection or partitions load state
	GetLoadState(ctx context.Context, collectionName string, partitionNames []string) (entity.LoadState, error)

	// ListResourceGroups returns list of resource group names in current Milvus instance.
	ListResourceGroups(ctx context.Context) ([]string, error)
	// CreateResourceGroup creates a resource group with provided name.
	CreateResourceGroup(ctx context.Context, rgName string) error
	// DescribeResourceGroup returns resource groups information.
	DescribeResourceGroup(ctx context.Context, rgName string) (*entity.ResourceGroup, error)
	// DropResourceGroup drops the resource group with provided name.
	DropResourceGroup(ctx context.Context, rgName string) error
	// TransferNode transfers querynodes between resource groups.
	TransferNode(ctx context.Context, sourceRg, targetRg string, nodesNum int32) error
	// TransferReplica transfer collection replicas between source,target resource group.
	TransferReplica(ctx context.Context, sourceRg, targetRg string, collectionName string, replicaNum int64) error

	// GetVersion get milvus version
	GetVersion(ctx context.Context) (string, error)
	// CheckHealth returns milvus state
	CheckHealth(ctx context.Context) (*entity.MilvusState, error)

	ReplicateMessage(ctx context.Context,
		channelName string, beginTs, endTs uint64,
		msgsBytes [][]byte, startPositions, endPositions []*msgpb.MsgPosition,
		opts ...ReplicateMessageOption,
	) (*entity.MessageInfo, error)

	HybridSearch(ctx context.Context, collName string, partitions []string, limit int, outputFields []string, reranker Reranker, subRequests []*ANNSearchRequest, opts ...SearchQueryOptionFunc) ([]SearchResult, error)
}

// NewClient create a client connected to remote milvus cluster.
// More connect option can be modified by Config.
func NewClient(ctx context.Context, config Config) (Client, error) {
	if err := config.parse(); err != nil {
		return nil, err
	}

	c := &GrpcClient{
		config: &config,
	}

	// Parse remote address.
	addr := c.config.getParsedAddress()

	// Parse grpc options
	options := c.config.getDialOption()

	// Connect the grpc server.
	if err := c.connect(ctx, addr, options...); err != nil {
		return nil, err
	}

	return c, nil
}

// NewGrpcClient create client with grpc addr
// the `Connect` API will be called for you
// dialOptions contains the dial option(s) that control the grpc dialing process
// !!!Deprecated in future, use `NewClient` first.
func NewGrpcClient(ctx context.Context, addr string, dialOptions ...grpc.DialOption) (Client, error) {
	return NewClient(ctx, Config{
		Address:     addr,
		DialOptions: dialOptions,
	})
}

// NewDefaultGrpcClient creates a new gRPC client.
// !!!Deprecated in future, use `NewClient` first.
func NewDefaultGrpcClient(ctx context.Context, addr string) (Client, error) {
	return NewClient(
		ctx, Config{
			Address: addr,
		},
	)
}

// NewDefaultGrpcClientWithURI creates a new gRPC client with URI.
// !!!Deprecated in future, use `NewClient` first.
func NewDefaultGrpcClientWithURI(ctx context.Context, uri, username, password string) (Client, error) {
	return NewClient(ctx, Config{
		Address:  uri,
		Username: username,
		Password: password,
	})
}

// NewDefaultGrpcClientWithTLSAuth creates a new gRPC client with TLS authentication.
// !!!Deprecated in future, use `NewClient` first.
func NewDefaultGrpcClientWithTLSAuth(ctx context.Context, addr, username, password string) (Client, error) {
	return NewClient(
		ctx,
		Config{
			Address:       addr,
			Username:      username,
			Password:      password,
			EnableTLSAuth: true,
		},
	)
}

// NewDefaultGrpcClientWithAuth creates a new gRPC client with authentication.
// !!!Deprecated in future, use `NewClient` first.
func NewDefaultGrpcClientWithAuth(ctx context.Context, addr, username, password string) (Client, error) {
	return NewClient(
		ctx,
		Config{
			Address:  addr,
			Username: username,
			Password: password,
		},
	)
}
