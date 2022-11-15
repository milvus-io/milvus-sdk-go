package base

import (
	"context"
	"log"

	server "github.com/milvus-io/milvus-proto/go-api/milvuspb"
	"github.com/milvus-io/milvus-sdk-go/v2/client"
	v2 "github.com/milvus-io/milvus-sdk-go/v2/client/v2"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"

	"google.golang.org/grpc"
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
	mClient v2.Client
}

func NewMilvusClient(ctx context.Context, addr string, dialOptions ...grpc.DialOption) (*MilvusClient, error) {
	preRequest("NewGrpcClient", addr, dialOptions)
	mClient, err := v2.NewGrpcClient(ctx, addr, dialOptions...)
	postResponse("NewGrpcClient", err, mClient)
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

// Create Collection
func (mc *MilvusClient) CreateCollection(ctx context.Context, collSchema *entity.Schema, shardsNum int32, opts ...client.CreateCollectionOption) error {
	preRequest("CreateCollection", collSchema, shardsNum, opts)
	err := mc.mClient.CreateCollection(ctx, collSchema, shardsNum, opts...)
	postResponse("CreateCollection", err)
	return err
}

// List Collections
func (mc *MilvusClient) ListCollections(ctx context.Context) ([]*entity.Collection, error) {
	preRequest("ListCollections", ctx)
	collections, err := mc.mClient.ListCollections(ctx)
	postResponse("ListCollections", err, collections)
	return collections, err
}

// Describe collection
func (mc *MilvusClient) DescribeCollection(ctx context.Context, collName string) (*entity.Collection, error) {
	preRequest("DescribeCollection", collName)
	collection, err := mc.mClient.DescribeCollection(ctx, collName)
	postResponse("DescribeCollection", err, collection)
	return collection, err
}

// Drop Collection
func (mc *MilvusClient) DropCollection(ctx context.Context, collName string) error {
	preRequest("DropCollection", collName)
	err := mc.mClient.DropCollection(ctx, collName)
	postResponse("DropCollection", err)
	return err
}

// Get Collection Statistics
func (mc *MilvusClient) GetCollectionStatistics(ctx context.Context, collName string) (map[string]string, error) {
	preRequest("GetCollectionStatistics", collName)
	stats, err := mc.mClient.GetCollectionStatistics(ctx, collName)
	postResponse("GetCollectionStatistics", err, stats)
	return stats, err
}

// Load Collection
func (mc *MilvusClient) LoadCollection(ctx context.Context, collName string, async bool, opts ...client.LoadCollectionOption) error {
	var funcName = "LoadCollection"
	preRequest(funcName, collName)
	err := mc.mClient.LoadCollection(ctx, collName, async, opts...)
	postResponse(funcName, err)
	return err
}

// Release Collection
func (mc *MilvusClient) ReleaseCollection(ctx context.Context, collName string) error {
	preRequest("ReleaseCollection", collName)
	err := mc.mClient.ReleaseCollection(ctx, collName)
	postResponse("ReleaseCollection", err)
	return err
}

// Show Partitions
func (mc *MilvusClient) ShowPartitions(ctx context.Context, collName string) ([]*entity.Partition, error) {
	preRequest("ShowPartitions", collName)
	partitions, err := mc.mClient.ShowPartitions(ctx, collName)
	postResponse("ShowPartitions", err, partitions)
	return partitions, err
}

// Get Persistent Segment Info
func (mc *MilvusClient) GetPersistentSegmentInfo(ctx context.Context, collName string) ([]*entity.Segment, error) {
	preRequest("GetPersistentSegmentInfo", collName)
	segments, err := mc.mClient.GetPersistentSegmentInfo(ctx, collName)
	postResponse("GetPersistentSegmentInfo", err, segments)
	return segments, err
}

// TODO collName, indexName, fieldName in CreateIndexRequestOption
// Create Index
func (mc *MilvusClient) CreateIndex(ctx context.Context, async bool, idx entity.Index, opts ...v2.CreateIndexRequestOption) error {
	preRequest("CreateIndex", async, idx, opts)
	err := mc.mClient.CreateIndex(ctx, async, idx, opts...)
	postResponse("CreateIndex", err)
	return err
}

// TODO milvuspb server IndexDescription
//Describe Index
func (mc *MilvusClient) DescribeIndex(ctx context.Context, opts ...v2.DescribeIndexRequestOption) ([]*server.IndexDescription, error) {
	preRequest("DescribeIndex", opts)
	indexes, err := mc.mClient.DescribeIndex(ctx, opts...)
	postResponse("DescribeIndex", err, indexes)
	return indexes, err
}

// Insert
func (mc *MilvusClient) Insert(ctx context.Context, collName string, partitionName string, columns ...entity.Column) (entity.Column, error) {
	preRequest("Insert", collName, partitionName, columns)
	ids, err := mc.mClient.Insert(ctx, collName, partitionName, columns...)
	postResponse("Insert", err, ids)
	return ids, err
}

// Flush
func (mc *MilvusClient) Flush(ctx context.Context, collName string, async bool) error {
	preRequest("Flush", collName, async)
	err := mc.mClient.Flush(ctx, collName, async)
	postResponse("Flush", err)
	return err
}

// Search
func (mc *MilvusClient) Search(ctx context.Context, collName string, partitions []string, expr string,
	outputFields []string, vectors []entity.Vector, vectorField string, metricType entity.MetricType, topK int, sp entity.SearchParam, opts ...client.SearchQueryOptionFunc) ([]client.SearchResult, error) {
	var funcName = "Search"
	preRequest(funcName, collName, partitions, expr, outputFields, vectors, vectorField, metricType, topK, sp, opts)

	searchResult, err := mc.mClient.Search(ctx, collName, partitions, expr, outputFields, vectors, vectorField, metricType, topK, sp, opts...)
	postResponse(funcName, err, searchResult)

	return searchResult, err
}

// Query
func (mc *MilvusClient) Query(ctx context.Context, collName string, partitions []string, ids entity.Column,
	outputFields []string, opts ...client.SearchQueryOptionFunc) ([]entity.Column, error) {
	var funcName = "QueryByPks"
	preRequest(funcName, collName, partitions, ids, outputFields, opts)

	queryResults, err := mc.mClient.QueryByPks(ctx, collName, partitions, ids, outputFields, opts...)

	postResponse(funcName, err, queryResults)
	return queryResults, err
}
