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

	"github.com/cockroachdb/errors"
	"github.com/golang/protobuf/proto"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
	"google.golang.org/grpc"

	common "github.com/milvus-io/milvus-proto/go-api/commonpb"
	server "github.com/milvus-io/milvus-proto/go-api/milvuspb"
)

// GrpcClient, uses default grpc Service definition to connect with Milvus2.0
type GrpcClient struct {
	Conn    *grpc.ClientConn           // grpc connection instance
	Service server.MilvusServiceClient // Service client stub
}

// connect connect to Service
func (c *GrpcClient) connect(ctx context.Context, addr string, opts ...grpc.DialOption) error {
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

// Close close the connection
func (c *GrpcClient) Close() error {
	if c.Conn != nil {
		err := c.Conn.Close()
		c.Conn = nil
		return err
	}
	return nil
}

// handles response status
// if status is nil returns ErrStatusNil
// if status.ErrorCode is common.ErrorCode_Success, returns nil
// otherwise, try use Reason into ErrServiceFailed
// if Reason is empty, returns ErrServiceFailed with default string
func handleRespStatus(status *common.Status) error {
	if status == nil {
		return ErrStatusNil
	}
	if status.ErrorCode != common.ErrorCode_Success {
		if status.GetReason() != "" {
			return ErrServiceFailed(errors.New(status.GetReason()))
		}
		return ErrServiceFailed(errors.New("Service failed"))
	}
	return nil
}

// ListCollections list collections from connection
// Note that schema info are not provided in collection list
func (c *GrpcClient) ListCollections(ctx context.Context) ([]*entity.Collection, error) {
	if c.Service == nil {
		return []*entity.Collection{}, ErrClientNotReady
	}
	req := &server.ShowCollectionsRequest{
		DbName:    "",
		TimeStamp: 0, // means now
	}
	resp, err := c.Service.ShowCollections(ctx, req)
	if err != nil {
		return []*entity.Collection{}, err
	}
	err = handleRespStatus(resp.GetStatus())
	if err != nil {
		return []*entity.Collection{}, err
	}
	collections := make([]*entity.Collection, 0, len(resp.GetCollectionIds()))
	for idx, item := range resp.CollectionIds {
		collection := &entity.Collection{
			ID:   item,
			Name: resp.GetCollectionNames()[idx],
		}
		if len(resp.GetInMemoryPercentages()) > idx {
			collection.Loaded = resp.GetInMemoryPercentages()[idx] == 100
		}
		collections = append(collections, collection)
	}
	return collections, nil
}

// CreateCollection create collection with specified schema
func (c *GrpcClient) CreateCollection(ctx context.Context, collSchema *entity.Schema, shardNum int32, opts ...CreateCollectionOption) error {
	if c.Service == nil {
		return ErrClientNotReady
	}
	if err := validateSchema(collSchema); err != nil {
		return err
	}

	has, err := c.HasCollection(ctx, collSchema.CollectionName)
	if err != nil {
		return err
	}
	if has {
		return fmt.Errorf("collection %s already exist", collSchema.CollectionName)
	}
	sch := collSchema.ProtoMessage()
	bs, err := proto.Marshal(sch)
	if err != nil {
		return err
	}
	req := &server.CreateCollectionRequest{
		DbName:         "", // reserved fields, not used for now
		CollectionName: collSchema.CollectionName,
		Schema:         bs,
		ShardsNum:      shardNum,
		// default consistency level is strong
		// to be consistent with previous version
		ConsistencyLevel: common.ConsistencyLevel_Strong,
	}
	// apply options on request
	for _, opt := range opts {
		opt(req)
	}
	resp, err := c.Service.CreateCollection(ctx, req)
	if err != nil {
		return err
	}
	err = handleRespStatus(resp)
	if err != nil {
		return err
	}
	return nil
}

func validateSchema(sch *entity.Schema) error {
	if sch == nil {
		return errors.New("nil schema")
	}
	if sch.CollectionName == "" {
		return errors.New("collection name cannot be empty")
	}

	primaryKey := false
	autoID := false
	vectors := 0
	for _, field := range sch.Fields {
		if field.PrimaryKey {
			if primaryKey { // another primary key found, only one primary key field for now
				return errors.New("only one primary key only")
			}
			if field.DataType != entity.FieldTypeInt64 && field.DataType != entity.FieldTypeVarChar { // string key not supported yet
				return errors.New("only int64 and varchar column can be primary key for now")
			}
			primaryKey = true
		}
		if field.AutoID {
			if autoID {
				return errors.New("only one auto id is available")
			}
			if field.DataType != entity.FieldTypeInt64 {
				return errors.New("only int64 column can be auto generated id")
			}
			autoID = true
		}
		if field.DataType == entity.FieldTypeFloatVector || field.DataType == entity.FieldTypeBinaryVector {
			vectors++
		}
	}
	if vectors <= 0 {
		return errors.New("vector field not set")
	}
	return nil
}

func (c *GrpcClient) checkCollectionExists(ctx context.Context, collName string) error {
	has, err := c.HasCollection(ctx, collName)
	if err != nil {
		return err
	}
	if !has {
		return collNotExistsErr(collName)
	}
	return nil
}

// DescribeCollection describe the collection by name
func (c *GrpcClient) DescribeCollection(ctx context.Context, collName string) (*entity.Collection, error) {
	if c.Service == nil {
		return nil, ErrClientNotReady
	}
	req := &server.DescribeCollectionRequest{
		CollectionName: collName,
	}
	resp, err := c.Service.DescribeCollection(ctx, req)
	if err != nil {
		return nil, err
	}
	err = handleRespStatus(resp.GetStatus())
	if err != nil {
		return nil, err
	}
	collection := &entity.Collection{
		ID:               resp.GetCollectionID(),
		Name:             collName,
		Schema:           (&entity.Schema{}).ReadProto(resp.GetSchema()),
		PhysicalChannels: resp.GetPhysicalChannelNames(),
		VirtualChannels:  resp.GetVirtualChannelNames(),
		ConsistencyLevel: entity.ConsistencyLevel(resp.ConsistencyLevel),
		ShardNum:         resp.GetShardsNum(),
	}
	collection.Name = collection.Schema.CollectionName
	colInfo := collInfo{
		ID:               collection.ID,
		Name:             collection.Name,
		Schema:           collection.Schema,
		ConsistencyLevel: collection.ConsistencyLevel,
	}
	MetaCache.setCollectionInfo(collName, &colInfo)
	return collection, nil
}

// DropCollection drop collection by name
func (c *GrpcClient) DropCollection(ctx context.Context, collName string) error {
	if c.Service == nil {
		return ErrClientNotReady
	}
	if err := c.checkCollectionExists(ctx, collName); err != nil {
		return err
	}

	req := &server.DropCollectionRequest{
		CollectionName: collName,
	}
	resp, err := c.Service.DropCollection(ctx, req)
	if err != nil {
		return err
	}
	err = handleRespStatus(resp)
	if err == nil {
		MetaCache.setCollectionInfo(collName, nil)
	}
	return err
}

// HasCollection check whether collection name exists
func (c *GrpcClient) HasCollection(ctx context.Context, collName string) (bool, error) {
	if c.Service == nil {
		return false, ErrClientNotReady
	}

	req := &server.HasCollectionRequest{
		DbName:         "", // reserved
		CollectionName: collName,
		TimeStamp:      0, // 0 for now
	}

	resp, err := c.Service.HasCollection(ctx, req)
	if err != nil {
		return false, err
	}
	if err := handleRespStatus(resp.GetStatus()); err != nil {
		return false, err
	}
	return resp.GetValue(), nil
}

// GetCollectionStatistcis show collection statistics
func (c *GrpcClient) GetCollectionStatistics(ctx context.Context, collName string) (map[string]string, error) {
	if c.Service == nil {
		return nil, ErrClientNotReady
	}

	if err := c.checkCollectionExists(ctx, collName); err != nil {
		return nil, err
	}

	req := &server.GetCollectionStatisticsRequest{
		CollectionName: collName,
	}
	resp, err := c.Service.GetCollectionStatistics(ctx, req)
	if err != nil {
		return nil, err
	}
	if err := handleRespStatus(resp.GetStatus()); err != nil {
		return nil, err
	}
	return entity.KvPairsMap(resp.GetStats()), nil
}

// ShowCollection show collection status, used to check whether it is loaded or not
func (c *GrpcClient) ShowCollection(ctx context.Context, collName string) (*entity.Collection, error) {
	if c.Service == nil {
		return nil, ErrClientNotReady
	}
	if err := c.checkCollectionExists(ctx, collName); err != nil {
		return nil, err
	}

	req := &server.ShowCollectionsRequest{
		Type:            server.ShowType_InMemory,
		CollectionNames: []string{collName},
	}

	resp, err := c.Service.ShowCollections(ctx, req)
	if err != nil {
		return nil, err
	}
	if err := handleRespStatus(resp.GetStatus()); err != nil {
		return nil, err
	}

	if len(resp.CollectionIds) != 1 || len(resp.InMemoryPercentages) != 1 {
		return nil, errors.New("response len not valid")
	}
	return &entity.Collection{
		ID:     resp.CollectionIds[0],
		Loaded: resp.InMemoryPercentages[0] == 100, // TODO silverxia, the percentage can be either 0 or 100
	}, nil
}

// LoadCollection load collection into memory
func (c *GrpcClient) LoadCollection(ctx context.Context, collName string, async bool, opts ...LoadCollectionOption) error {
	if c.Service == nil {
		return ErrClientNotReady
	}

	if err := c.checkCollectionExists(ctx, collName); err != nil {
		return err
	}

	req := &server.LoadCollectionRequest{
		CollectionName: collName,
		ReplicaNumber:  1, // default replica number
	}

	for _, opt := range opts {
		opt(req)
	}

	resp, err := c.Service.LoadCollection(ctx, req)
	if err != nil {
		return err
	}
	if err := handleRespStatus(resp); err != nil {
		return err
	}

	if !async {
		for {
			select {
			case <-ctx.Done():
				return errors.New("context deadline exceeded")
			default:
			}

			coll, err := c.ShowCollection(ctx, collName)
			if err != nil {
				return err
			}
			if coll.Loaded {
				break
			}

			time.Sleep(200 * time.Millisecond) // TODO change to configuration
		}
	}
	return nil
}

// ReleaseCollection release loaded collection
func (c *GrpcClient) ReleaseCollection(ctx context.Context, collName string) error {
	if c.Service == nil {
		return ErrClientNotReady
	}
	if err := c.checkCollectionExists(ctx, collName); err != nil {
		return err
	}

	req := &server.ReleaseCollectionRequest{
		DbName:         "", // reserved
		CollectionName: collName,
	}
	resp, err := c.Service.ReleaseCollection(ctx, req)
	if err != nil {
		return err
	}
	return handleRespStatus(resp)
}

// GetReplicas gets the replica groups as well as their querynodes and shards information
func (c *GrpcClient) GetReplicas(ctx context.Context, collName string) ([]*entity.ReplicaGroup, error) {
	if c.Service == nil {
		return nil, ErrClientNotReady
	}
	coll, err := c.ShowCollection(ctx, collName)
	if err != nil {
		return nil, err
	}

	req := &server.GetReplicasRequest{
		CollectionID:   coll.ID,
		WithShardNodes: true, // return nodes by default
	}

	resp, err := c.Service.GetReplicas(ctx, req)
	if err != nil {
		return nil, err
	}
	if err = handleRespStatus(resp.GetStatus()); err != nil {
		return nil, err
	}

	groups := make([]*entity.ReplicaGroup, 0, len(resp.GetReplicas()))
	for _, rp := range resp.GetReplicas() {
		group := &entity.ReplicaGroup{
			ReplicaID:     rp.ReplicaID,
			NodeIDs:       rp.NodeIds,
			ShardReplicas: make([]*entity.ShardReplica, 0, len(rp.ShardReplicas)),
		}
		for _, s := range rp.ShardReplicas {
			shard := &entity.ShardReplica{
				LeaderID:      s.LeaderID,
				NodesIDs:      s.NodeIds,
				DmChannelName: s.DmChannelName,
			}
			group.ShardReplicas = append(group.ShardReplicas, shard)
		}
		groups = append(groups, group)
	}
	return groups, nil
}

// GetLoadingProgress get the collection or partitions loading progress
func (c *GrpcClient) GetLoadingProgress(ctx context.Context, collName string, partitionNames []string) (int64, error) {
	if c.Service == nil {
		return 0, ErrClientNotReady
	}
	if err := c.checkCollectionExists(ctx, collName); err != nil {
		return 0, err
	}

	req := &server.GetLoadingProgressRequest{
		CollectionName: collName,
		PartitionNames: partitionNames,
	}
	resp, err := c.Service.GetLoadingProgress(ctx, req)
	if err != nil {
		return 0, err
	}

	return resp.GetProgress(), nil
}

// GetLoadState get the collection or partitions load state
func (c *GrpcClient) GetLoadState(ctx context.Context, collName string, partitionNames []string) (entity.LoadState, error) {
	if c.Service == nil {
		return 0, ErrClientNotReady
	}
	if err := c.checkCollectionExists(ctx, collName); err != nil {
		return 0, err
	}

	req := &server.GetLoadStateRequest{
		CollectionName: collName,
		PartitionNames: partitionNames,
	}
	resp, err := c.Service.GetLoadState(ctx, req)
	if err != nil {
		return 0, err
	}

	return entity.LoadState(resp.GetState()), nil
}

// AlterCollection changes the collection attribute.
func (c *GrpcClient) AlterCollection(ctx context.Context, collName string, attrs ...entity.CollectionAttribute) error {
	if c.Service == nil {
		return ErrClientNotReady
	}
	if err := c.checkCollectionExists(ctx, collName); err != nil {
		return err
	}

	if len(attrs) == 0 {
		return errors.New("no collection attribute provided")
	}

	keys := make(map[string]struct{})

	props := make([]*common.KeyValuePair, 0, len(attrs))
	for _, attr := range attrs {
		k, v := attr.KeyValue()
		if _, exists := keys[k]; exists {
			return errors.New("duplicated attributed received")
		}
		keys[k] = struct{}{}
		props = append(props, &common.KeyValuePair{
			Key:   k,
			Value: v,
		})
	}

	req := &server.AlterCollectionRequest{
		CollectionName: collName,
		Properties:     props,
	}

	resp, err := c.Service.AlterCollection(ctx, req)
	if err != nil {
		return err
	}
	return handleRespStatus(resp)
}
