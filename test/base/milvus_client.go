package base

import (
	"context"
	"fmt"
	"log"
	"time"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
)

func preRequest(funcName string, args ...interface{}) {
	log.Printf("(ApiRequest): func [%s], args: %v\n", funcName, args)
}

func postResponse(funcName string, err error, res ...interface{}) {
	if err != nil {
		log.Printf("(ApiResponse): func [%s], error: %s\n", funcName, err)
	} else {
		log.Printf("(ApiResponse): func [%s], results: %v\n", funcName, res)
	}
}

type MilvusClient struct {
	mClient client.Client
}

func NewMilvusClient(ctx context.Context, cfg client.Config) (*MilvusClient, error) {
	preRequest("NewClient", ctx, cfg)
	mClient, err := client.NewClient(ctx, cfg)
	postResponse("NewClient", err, mClient)
	return &MilvusClient{
		mClient,
	}, err
}

// !!!Deprecated in future, use `NewMilvusClient` first.
func NewDefaultMilvusClient(ctx context.Context, addr string) (*MilvusClient, error) {
	preRequest("NewDefaultGrpcClient", ctx, addr)
	mClient, err := client.NewDefaultGrpcClient(ctx, addr)
	postResponse("NewDefaultGrpcClient", err, mClient)
	return &MilvusClient{
		mClient,
	}, err
}

func NewDefaultMilvusClientWithURI(ctx context.Context, uri, username, password string) (*MilvusClient, error) {
	preRequest("NewDefaultGrpcClientWithURI", ctx, uri, username)
	mClient, err := client.NewDefaultGrpcClientWithURI(ctx, uri, username, password)
	postResponse("NewDefaultGrpcClientWithURI", err, mClient)
	return &MilvusClient{
		mClient,
	}, err
}

func NewDefaultMilvusClientWithTLSAuth(ctx context.Context, addr, username, password string) (*MilvusClient, error) {
	preRequest("NewDefaultGrpcClientWithTLSAuth", ctx, addr, username)
	mClient, err := client.NewDefaultGrpcClientWithTLSAuth(ctx, addr, username, password)
	postResponse("NewDefaultGrpcClientWithTLSAuth", err, mClient)
	return &MilvusClient{
		mClient,
	}, err
}

func NewDefaultMilvusClientWithAuth(ctx context.Context, addr, username, password string) (*MilvusClient, error) {
	preRequest("NewDefaultGrpcClientWithAuth", ctx, addr, username)
	mClient, err := client.NewDefaultGrpcClientWithAuth(ctx, addr, username, password)
	postResponse("NewDefaultGrpcClientWithAuth", err, mClient)
	return &MilvusClient{
		mClient,
	}, err
}

func (mc *MilvusClient) Close() error {
	preRequest("Close")
	err := mc.mClient.Close()
	postResponse("Close", err)
	return err
}

// -- database --

// UsingDatabase for database operation after this function call.
// All request in any goroutine will be applied to new database on the same client. e.g.
// 1. goroutine A access DB1.
// 2. goroutine B call UsingDatabase(ctx, "DB2").
// 3. goroutine A access DB2 after 2.
func (mc *MilvusClient) UsingDatabase(ctx context.Context, dbName string) error {
	preRequest("UsingDatabase", ctx, dbName)
	err := mc.mClient.UsingDatabase(ctx, dbName)
	postResponse("UsingDatabase", err)
	return err
}

// -- database --

// ListDatabases list all database in milvus cluster.
func (mc *MilvusClient) ListDatabases(ctx context.Context) ([]entity.Database, error) {
	preRequest("ListDatabases", ctx)
	dbs, err := mc.mClient.ListDatabases(ctx)
	postResponse("ListDatabases", err, dbs)
	return dbs, err
}

// CreateDatabase create database with the given name.
func (mc *MilvusClient) CreateDatabase(ctx context.Context, dbName string) error {
	preRequest("CreateDatabase", ctx, dbName)
	err := mc.mClient.CreateDatabase(ctx, dbName)
	postResponse("CreateDatabase", err)
	return err
}

// DropDatabase drop database with the given db name.
func (mc *MilvusClient) DropDatabase(ctx context.Context, dbName string) error {
	preRequest("DropDatabase", ctx, dbName)
	err := mc.mClient.DropDatabase(ctx, dbName)
	postResponse("DropDatabase", err)
	return err
}

// -- collection --

// CreateCollection Create Collection
func (mc *MilvusClient) CreateCollection(ctx context.Context, collSchema *entity.Schema, shardsNum int32, opts ...client.CreateCollectionOption) error {
	if collSchema == nil {
		preRequest("CreateCollection", ctx, collSchema, shardsNum, opts)
	} else {
		preRequest("CreateCollection", ctx, collSchema.CollectionName, collSchema, shardsNum, opts)
	}
	err := mc.mClient.CreateCollection(ctx, collSchema, shardsNum, opts...)
	postResponse("CreateCollection", err)
	return err
}

// NewCollection highlevel new Collection
func (mc *MilvusClient) NewCollection(ctx context.Context, collName string, dim int64, opts ...client.CreateCollectionOption) error {
	preRequest("NewCollection", ctx, collName, dim, opts)
	err := mc.mClient.NewCollection(ctx, collName, dim, opts...)
	postResponse("NewCollection", err)
	return err
}

// ListCollections list collections
func (mc *MilvusClient) ListCollections(ctx context.Context) ([]*entity.Collection, error) {
	preRequest("ListCollections", ctx)
	collections, err := mc.mClient.ListCollections(ctx)
	postResponse("ListCollections", err, collections)
	return collections, err
}

// DescribeCollection Describe collection
func (mc *MilvusClient) DescribeCollection(ctx context.Context, collName string) (*entity.Collection, error) {
	preRequest("DescribeCollection", ctx, collName)
	collection, err := mc.mClient.DescribeCollection(ctx, collName)
	postResponse("DescribeCollection", err, collection)
	return collection, err
}

// DropCollection Drop Collection
func (mc *MilvusClient) DropCollection(ctx context.Context, collName string) error {
	preRequest("DropCollection", ctx, collName)
	err := mc.mClient.DropCollection(ctx, collName)
	postResponse("DropCollection", err)
	return err
}

// GetCollectionStatistics Get Collection Statistics
func (mc *MilvusClient) GetCollectionStatistics(ctx context.Context, collName string) (map[string]string, error) {
	preRequest("GetCollectionStatistics", ctx, collName)
	stats, err := mc.mClient.GetCollectionStatistics(ctx, collName)
	postResponse("GetCollectionStatistics", err, stats)
	return stats, err
}

// LoadCollection Load Collection
func (mc *MilvusClient) LoadCollection(ctx context.Context, collName string, async bool, opts ...client.LoadCollectionOption) error {
	funcName := "LoadCollection"
	preRequest(funcName, ctx, collName, opts)
	err := mc.mClient.LoadCollection(ctx, collName, async, opts...)
	postResponse(funcName, err)
	return err
}

// ReleaseCollection Release Collection
func (mc *MilvusClient) ReleaseCollection(ctx context.Context, collName string) error {
	preRequest("ReleaseCollection", ctx, collName)
	err := mc.mClient.ReleaseCollection(ctx, collName)
	postResponse("ReleaseCollection", err)
	return err
}

// HasCollection Has Collection
func (mc *MilvusClient) HasCollection(ctx context.Context, collName string) (bool, error) {
	preRequest("HasCollection", ctx, collName)
	has, err := mc.mClient.HasCollection(ctx, collName)
	postResponse("HasCollection", err, has)
	return has, err
}

// -- alias --

// CreateAlias Create Alias
func (mc *MilvusClient) CreateAlias(ctx context.Context, collName string, alias string) error {
	preRequest("CreateAlias", ctx, collName, alias)
	err := mc.mClient.CreateAlias(ctx, collName, alias)
	postResponse("CreateAlias", err)
	return err
}

// DropAlias drop alias
func (mc *MilvusClient) DropAlias(ctx context.Context, alias string) error {
	preRequest("DropAlias", ctx, alias)
	err := mc.mClient.DropAlias(ctx, alias)
	postResponse("DropAlias", err)
	return err
}

// AlterAlias Alter Alias
func (mc *MilvusClient) AlterAlias(ctx context.Context, collName string, alias string) error {
	preRequest("AlterAlias", ctx, collName, alias)
	err := mc.mClient.AlterAlias(ctx, collName, alias)
	postResponse("AlterAlias", err)
	return err
}

// GetReplicas Get Replicas
func (mc *MilvusClient) GetReplicas(ctx context.Context, collName string) ([]*entity.ReplicaGroup, error) {
	preRequest("GetReplicas", ctx, collName)
	replicas, err := mc.mClient.GetReplicas(ctx, collName)
	postResponse("GetReplicas", err, replicas)
	return replicas, err
}

// -- authentication --

// CreateCredential Create Credential
func (mc *MilvusClient) CreateCredential(ctx context.Context, username string, password string) error {
	preRequest("CreateCredential", ctx, username)
	err := mc.mClient.CreateCredential(ctx, username, password)
	postResponse("CreateCredential", err)
	return err
}

// UpdateCredential Update Credential
func (mc *MilvusClient) UpdateCredential(ctx context.Context, username string, oldPassword string, newPassword string) error {
	preRequest("UpdateCredential", ctx, username)
	err := mc.mClient.UpdateCredential(ctx, username, oldPassword, newPassword)
	postResponse("UpdateCredential", err)
	return err
}

// DeleteCredential delete credential
func (mc *MilvusClient) DeleteCredential(ctx context.Context, username string) error {
	preRequest("DeleteCredential", ctx, username)
	err := mc.mClient.DeleteCredential(ctx, username)
	postResponse("DeleteCredential", err)
	return err
}

// ListCredUsers list all usernames
func (mc *MilvusClient) ListCredUsers(ctx context.Context) ([]string, error) {
	preRequest("ListCredUsers", ctx)
	users, err := mc.mClient.ListCredUsers(ctx)
	postResponse("ListCredUsers", err, users)
	return users, err
}

// -- partition --

// CreatePartition Create Partition
func (mc *MilvusClient) CreatePartition(ctx context.Context, collName string, partitionName string) error {
	preRequest("CreatePartition", ctx, collName, partitionName)
	err := mc.mClient.CreatePartition(ctx, collName, partitionName)
	postResponse("CreatePartition", err)
	return err
}

// DropPartition Drop Partition
func (mc *MilvusClient) DropPartition(ctx context.Context, collName string, partitionName string) error {
	preRequest("DropPartition", ctx, collName, partitionName)
	err := mc.mClient.DropPartition(ctx, collName, partitionName)
	postResponse("DropPartition", err)
	return err
}

// ShowPartitions Show Partitions
func (mc *MilvusClient) ShowPartitions(ctx context.Context, collName string) ([]*entity.Partition, error) {
	preRequest("ShowPartitions", ctx, collName)
	partitions, err := mc.mClient.ShowPartitions(ctx, collName)
	postResponse("ShowPartitions", err, partitions)
	return partitions, err
}

// HasPartition Has Partition
func (mc *MilvusClient) HasPartition(ctx context.Context, collName string, partitionName string) (bool, error) {
	preRequest("HasPartition", ctx, collName)
	has, err := mc.mClient.HasPartition(ctx, collName, partitionName)
	postResponse("HasPartition", err, has)
	return has, err
}

// LoadPartitions Load Partitions into memory
func (mc *MilvusClient) LoadPartitions(ctx context.Context, collName string, partitionNames []string, async bool) error {
	preRequest("LoadPartitions", ctx, collName, partitionNames, async)
	err := mc.mClient.LoadPartitions(ctx, collName, partitionNames, async)
	postResponse("LoadPartitions", err)
	return err
}

// ReleasePartitions release partitions
func (mc *MilvusClient) ReleasePartitions(ctx context.Context, collName string, partitionNames []string) error {
	preRequest("ReleasePartitions", ctx, collName, partitionNames)
	err := mc.mClient.ReleasePartitions(ctx, collName, partitionNames)
	postResponse("ReleasePartitions", err)
	return err
}

// GetPersistentSegmentInfo Get Persistent Segment Info
func (mc *MilvusClient) GetPersistentSegmentInfo(ctx context.Context, collName string) ([]*entity.Segment, error) {
	preRequest("GetPersistentSegmentInfo", ctx, collName)
	segments, err := mc.mClient.GetPersistentSegmentInfo(ctx, collName)
	postResponse("GetPersistentSegmentInfo", err, segments)
	return segments, err
}

// CreateIndex Create Index
func (mc *MilvusClient) CreateIndex(ctx context.Context, collName string, fieldName string, idx entity.Index, async bool, opts ...client.IndexOption) error {
	preRequest("CreateIndex", ctx, collName, fieldName, async, idx, opts)
	err := mc.mClient.CreateIndex(ctx, collName, fieldName, idx, async, opts...)
	postResponse("CreateIndex", err)
	return err
}

// DescribeIndex Describe Index
func (mc *MilvusClient) DescribeIndex(ctx context.Context, collectionName string, fieldName string, opts ...client.IndexOption) ([]entity.Index, error) {
	preRequest("DescribeIndex", ctx, collectionName, fieldName, opts)
	indexes, err := mc.mClient.DescribeIndex(ctx, collectionName, fieldName, opts...)
	postResponse("DescribeIndex", err, indexes)
	return indexes, err
}

// DropIndex Drop Index
func (mc *MilvusClient) DropIndex(ctx context.Context, collName string, fieldName string, opts ...client.IndexOption) error {
	preRequest("DropIndex", ctx, collName, fieldName, opts)
	err := mc.mClient.DropIndex(ctx, collName, fieldName, opts...)
	postResponse("DropIndex", err)
	return err
}

// GetIndexState Get IndexState, index naming is not supported yet
func (mc *MilvusClient) GetIndexState(ctx context.Context, collName string, fieldName string, opts ...client.IndexOption) (entity.IndexState, error) {
	preRequest("GetIndexState", ctx, collName, fieldName, opts)
	indexState, err := mc.mClient.GetIndexState(ctx, collName, fieldName, opts...)
	postResponse("GetIndexState", err, indexState)
	return indexState, err
}

// -- basic operation --

// Insert insert data
func (mc *MilvusClient) Insert(ctx context.Context, collName string, partitionName string, columns ...entity.Column) (entity.Column, error) {
	preRequest("Insert", ctx, collName, partitionName, columns)
	ids, err := mc.mClient.Insert(ctx, collName, partitionName, columns...)
	postResponse("Insert", err, ids)
	return ids, err
}

// Flush flush collection
func (mc *MilvusClient) Flush(ctx context.Context, collName string, async bool) error {
	preRequest("Flush", ctx, collName, async)
	err := mc.mClient.Flush(ctx, collName, async)
	postResponse("Flush", err)
	return err
}

// DeleteByPks deletes entries related to provided primary keys
func (mc *MilvusClient) DeleteByPks(ctx context.Context, collName string, partitionName string, ids entity.Column) error {
	preRequest("DeleteByPks", ctx, collName, partitionName, ids)
	err := mc.mClient.DeleteByPks(ctx, collName, partitionName, ids)
	postResponse("DeleteByPks", err)
	return err
}

// Delete deletes entries match expression
func (mc *MilvusClient) Delete(ctx context.Context, collName string, partitionName string, expr string) error {
	preRequest("Delete", ctx, collName, partitionName, expr)
	err := mc.mClient.Delete(ctx, collName, partitionName, expr)
	postResponse("Delete", err)
	return err
}

func (mc *MilvusClient) Upsert(ctx context.Context, collName string, partitionName string, columns ...entity.Column) (entity.Column, error) {
	preRequest("Upsert", ctx, collName, partitionName, columns)
	ids, err := mc.mClient.Upsert(ctx, collName, partitionName, columns...)
	postResponse("Upsert", err, ids)
	return ids, err
}

// Search search from collection
func (mc *MilvusClient) Search(ctx context.Context, collName string, partitions []string, expr string,
	outputFields []string, vectors []entity.Vector, vectorField string, metricType entity.MetricType, topK int, sp entity.SearchParam, opts ...client.SearchQueryOptionFunc,
) ([]client.SearchResult, error) {
	funcName := "Search"
	preRequest(funcName, ctx, collName, partitions, expr, outputFields, fmt.Sprintf("nq=%d", len(vectors)), vectorField, metricType, topK, sp, opts)

	searchResult, err := mc.mClient.Search(ctx, collName, partitions, expr, outputFields, vectors, vectorField, metricType, topK, sp, opts...)
	postResponse(funcName, err, searchResult)

	return searchResult, err
}

func (mc *MilvusClient) HybridSearch(ctx context.Context, collName string, partitions []string, limit int, outputFields []string,
	reranker client.Reranker, subRequests []*client.ANNSearchRequest) ([]client.SearchResult, error) {
	funcName := "HybridSearch"
	preRequest(funcName, ctx, collName, partitions, limit, outputFields, reranker, subRequests)

	searchResult, err := mc.mClient.HybridSearch(ctx, collName, partitions, limit, outputFields, reranker, subRequests)
	postResponse(funcName, err, searchResult)

	return searchResult, err
}

// QueryByPks query from collection
func (mc *MilvusClient) QueryByPks(ctx context.Context, collName string, partitions []string, ids entity.Column,
	outputFields []string, opts ...client.SearchQueryOptionFunc) (client.ResultSet, error) {
	funcName := "QueryByPks"
	preRequest(funcName, ctx, collName, partitions, ids, outputFields, opts)

	queryResults, err := mc.mClient.QueryByPks(ctx, collName, partitions, ids, outputFields, opts...)

	postResponse(funcName, err, queryResults)
	return queryResults, err
}

// Query query from collection
func (mc *MilvusClient) Query(ctx context.Context, collName string, partitions []string, expr string, outputFields []string, opts ...client.SearchQueryOptionFunc,
) (client.ResultSet, error) {
	funcName := "Query"
	preRequest(funcName, ctx, collName, partitions, expr, outputFields, opts)

	queryResults, err := mc.mClient.Query(ctx, collName, partitions, expr, outputFields, opts...)

	postResponse(funcName, err, queryResults)
	return queryResults, err
}

// Get query from collection
func (mc *MilvusClient) Get(ctx context.Context, collName string, ids entity.Column, opts ...client.GetOption,
) (client.ResultSet, error) {
	funcName := "Get"
	preRequest(funcName, ctx, collName, ids, opts)
	queryResults, err := mc.mClient.Get(ctx, collName, ids, opts...)
	postResponse(funcName, err, queryResults)
	return queryResults, err
}

// -- row based apis --

// CreateCollectionByRow Create Collection By Row
func (mc *MilvusClient) CreateCollectionByRow(ctx context.Context, row entity.Row, shardNum int32) error {
	preRequest("CreateCollectionByRow", ctx, row, shardNum)
	err := mc.mClient.CreateCollectionByRow(ctx, row, shardNum)
	postResponse("CreateCollectionByRow", err)
	return err
}

// InsertRows InsertByRows insert by rows
func (mc *MilvusClient) InsertRows(ctx context.Context, collName string, partitionName string, rows []interface{}) (entity.Column, error) {
	preRequest("InsertRows", ctx, collName, partitionName, len(rows))
	column, err := mc.mClient.InsertRows(ctx, collName, partitionName, rows)
	postResponse("InsertRows", err, column)
	return column, err
}

// Compact Manual Compaction
func (mc *MilvusClient) Compact(ctx context.Context, collName string, toleranceDuration time.Duration) (int64, error) {
	preRequest("ManualCompaction", ctx, collName, toleranceDuration)
	compactionID, err := mc.mClient.ManualCompaction(ctx, collName, toleranceDuration)
	postResponse("ManualCompaction", err, compactionID)
	return compactionID, err
}

// GetCompactionState Get Compaction State
func (mc *MilvusClient) GetCompactionState(ctx context.Context, id int64) (entity.CompactionState, error) {
	preRequest("GetCompactionState", ctx, id)
	compactionState, err := mc.mClient.GetCompactionState(ctx, id)
	postResponse("GetCompactionState", err, compactionState)
	return compactionState, err
}

// GetCompactionStateWithPlans Get Compaction State With Plans
func (mc *MilvusClient) GetCompactionStateWithPlans(ctx context.Context, id int64) (entity.CompactionState, []entity.CompactionPlan, error) {
	preRequest("GetCompactionStateWithPlans", ctx, id)
	compactionState, compactionPlan, err := mc.mClient.GetCompactionStateWithPlans(ctx, id)
	postResponse("GetCompactionStateWithPlans", err, compactionState, compactionPlan)
	return compactionState, compactionPlan, err
}

// BulkInsert Bulk Insert import data files(json, numpy, etc.) on MinIO/S3 storage, read and parse them into sealed segments
func (mc *MilvusClient) BulkInsert(ctx context.Context, collName string, partitionName string, files []string, opts ...client.BulkInsertOption) (int64, error) {
	preRequest("BulkInsert", ctx, collName, partitionName, files, opts)
	taskID, err := mc.mClient.BulkInsert(ctx, collName, partitionName, files, opts...)
	postResponse("BulkInsert", err, taskID)
	return taskID, err
}

// GetBulkInsertState checks import task state
func (mc *MilvusClient) GetBulkInsertState(ctx context.Context, taskID int64) (*entity.BulkInsertTaskState, error) {
	preRequest("GetBulkInsertState", ctx, taskID)
	bulkInsertTaskState, err := mc.mClient.GetBulkInsertState(ctx, taskID)
	postResponse("GetBulkInsertState", err, bulkInsertTaskState)
	return bulkInsertTaskState, err
}

// ListBulkInsertTasks List Bulk Insert Tasks
func (mc *MilvusClient) ListBulkInsertTasks(ctx context.Context, collName string, limit int64) ([]*entity.BulkInsertTaskState, error) {
	preRequest("ListBulkInsertTasks", ctx, collName, limit)
	bulkInsertTaskStates, err := mc.mClient.ListBulkInsertTasks(ctx, collName, limit)
	postResponse("ListBulkInsertTasks", err, bulkInsertTaskStates)
	return bulkInsertTaskStates, err
}

// GetLoadingProgress
func (mc *MilvusClient) GetLoadingProgress(ctx context.Context, collName string, partitionNames []string) (int64, error) {
	preRequest("GetLoadingProgress", ctx, collName, partitionNames)
	loadingProgress, err := mc.mClient.GetLoadingProgress(ctx, collName, partitionNames)
	postResponse("GetLoadingProgress", err, loadingProgress)
	return loadingProgress, err
}

// GetLoadState
func (mc *MilvusClient) GetLoadState(ctx context.Context, collName string, partitionNames []string) (entity.LoadState, error) {
	preRequest("GetLoadState", ctx, collName, partitionNames)
	loadState, err := mc.mClient.GetLoadState(ctx, collName, partitionNames)
	postResponse("GetLoadState", err, loadState)
	return loadState, err
}

// *** self-hosted
func (mc *MilvusClient) WaitForCompactionCompleted(ctx context.Context, compactionID int64) error {
	for {
		time.Sleep(time.Millisecond * 500)
		compactionState, err := mc.GetCompactionState(ctx, compactionID)
		if err != nil {
			return err
		}
		if compactionState == entity.CompactionStateCompleted {
			return nil
		}
	}
}

// ListResourceGroups List Resource Groups
func (mc *MilvusClient) ListResourceGroups(ctx context.Context) ([]string, error) {
	preRequest("ListResourceGroups", ctx)
	rgs, err := mc.mClient.ListResourceGroups(ctx)
	postResponse("ListResourceGroups", err, rgs)
	return rgs, err
}

// CreateResourceGroup
func (mc *MilvusClient) CreateResourceGroup(ctx context.Context, rgName string) error {
	preRequest("CreateResourceGroup", ctx, rgName)
	err := mc.mClient.CreateResourceGroup(ctx, rgName)
	postResponse("CreateResourceGroup", err)
	return err
}

// DescribeResourceGroup describe resource group
func (mc *MilvusClient) DescribeResourceGroup(ctx context.Context, rgName string) (*entity.ResourceGroup, error) {
	preRequest("DescribeResourceGroup", ctx, rgName)
	rg, err := mc.mClient.DescribeResourceGroup(ctx, rgName)
	postResponse("DescribeResourceGroup", err, rg)
	return rg, err
}

// DropResourceGroup drop resource group
func (mc *MilvusClient) DropResourceGroup(ctx context.Context, rgName string) error {
	preRequest("DropResourceGroup", ctx, rgName)
	err := mc.mClient.DropResourceGroup(ctx, rgName)
	postResponse("DropResourceGroup", err)
	return err
}

// TransferNode transfer node
func (mc *MilvusClient) TransferNode(ctx context.Context, sourceRg, targetRg string, nodesNum int32) error {
	preRequest("TransferNode", ctx, sourceRg, targetRg, nodesNum)
	err := mc.mClient.TransferNode(ctx, sourceRg, targetRg, nodesNum)
	postResponse("TransferNode", err)
	return err
}

// TransferReplica transfer replica
func (mc *MilvusClient) TransferReplica(ctx context.Context, sourceRg, targetRg string, collectionName string, replicaNum int64) error {
	preRequest("TransferReplica", ctx, sourceRg, targetRg, collectionName, replicaNum)
	err := mc.mClient.TransferReplica(ctx, sourceRg, targetRg, collectionName, replicaNum)
	postResponse("TransferReplica", err)
	return err
}
