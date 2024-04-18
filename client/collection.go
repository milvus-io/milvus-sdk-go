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
	"time"

	"github.com/cockroachdb/errors"

	"github.com/golang/protobuf/proto"

	"github.com/milvus-io/milvus-sdk-go/v2/entity"

	"github.com/milvus-io/milvus-proto/go-api/v2/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v2/milvuspb"
)

// handles response status
// if status is nil returns ErrStatusNil
// if status.ErrorCode is commonpb.ErrorCode_Success, returns nil
// otherwise, try use Reason into ErrServiceFailed
// if Reason is empty, returns ErrServiceFailed with default string
func handleRespStatus(status *commonpb.Status) error {
	if status == nil {
		return ErrStatusNil
	}
	if status.ErrorCode != commonpb.ErrorCode_Success {
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
	req := &milvuspb.ShowCollectionsRequest{
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

// NewCollection creates a common simple collection with pre-defined attributes.
func (c *GrpcClient) NewCollection(ctx context.Context, collName string, dimension int64, opts ...CreateCollectionOption) error {
	if c.Service == nil {
		return ErrClientNotReady
	}

	//	shardNum := entity.DefaultShardNumber
	opt := &createCollOpt{
		ConsistencyLevel:    entity.DefaultConsistencyLevel,
		PrimaryKeyFieldName: "id",
		PrimaryKeyFieldType: entity.FieldTypeInt64,
		VectorFieldName:     "vector",
		MetricsType:         entity.IP,
		AutoID:              false,
		EnableDynamicSchema: true,
	}

	for _, o := range opts {
		o(opt)
	}

	pkField := entity.NewField().WithName(opt.PrimaryKeyFieldName).WithDataType(opt.PrimaryKeyFieldType).WithIsAutoID(opt.AutoID).WithIsPrimaryKey(true)
	if opt.PrimaryKeyFieldType == entity.FieldTypeVarChar && opt.PrimaryKeyMaxLength > 0 {
		pkField = pkField.WithMaxLength(opt.PrimaryKeyMaxLength)
	}

	sch := entity.NewSchema().WithName(collName).WithAutoID(opt.AutoID).WithDynamicFieldEnabled(opt.EnableDynamicSchema).
		WithField(pkField).
		WithField(entity.NewField().WithName(opt.VectorFieldName).WithDataType(entity.FieldTypeFloatVector).WithDim(dimension))

	if err := c.validateSchema(sch); err != nil {
		return err
	}

	if err := c.requestCreateCollection(ctx, sch, opt, entity.DefaultShardNumber); err != nil {
		return err
	}

	idx := entity.NewGenericIndex("", "", map[string]string{
		"metric_type": string(opt.MetricsType),
	})

	if err := c.CreateIndex(ctx, collName, opt.VectorFieldName, idx, false); err != nil {
		return err
	}

	return c.LoadCollection(ctx, collName, false)
}

// CreateCollection create collection with specified schema
func (c *GrpcClient) CreateCollection(ctx context.Context, collSchema *entity.Schema, shardNum int32, opts ...CreateCollectionOption) error {
	if c.Service == nil {
		return ErrClientNotReady
	}
	if err := c.validateSchema(collSchema); err != nil {
		return err
	}

	opt := &createCollOpt{
		ConsistencyLevel: entity.DefaultConsistencyLevel,
		NumPartitions:    0,
	}
	// apply options on request
	for _, o := range opts {
		o(opt)
	}

	return c.requestCreateCollection(ctx, collSchema, opt, shardNum)
}

func (c *GrpcClient) requestCreateCollection(ctx context.Context, sch *entity.Schema, opt *createCollOpt, shardNum int32) error {
	if opt.EnableDynamicSchema {
		sch.EnableDynamicField = true
	}
	bs, err := proto.Marshal(sch.ProtoMessage())
	if err != nil {
		return err
	}

	req := &milvuspb.CreateCollectionRequest{
		Base:             opt.MsgBase,
		DbName:           "", // reserved fields, not used for now
		CollectionName:   sch.CollectionName,
		Schema:           bs,
		ShardsNum:        shardNum,
		ConsistencyLevel: opt.ConsistencyLevel.CommonConsistencyLevel(),
		NumPartitions:    opt.NumPartitions,
		Properties:       entity.MapKvPairs(opt.Properties),
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

func (c *GrpcClient) validateSchema(sch *entity.Schema) error {
	if sch == nil {
		return errors.New("nil schema")
	}
	if sch.CollectionName == "" {
		return errors.New("collection name cannot be empty")
	}

	primaryKey := false
	autoID := false
	vectors := 0
	hasPartitionKey := false
	hasDynamicSchema := sch.EnableDynamicField
	hasJSON := false
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
			autoID = true
		}
		if field.DataType == entity.FieldTypeJSON {
			hasJSON = true
		}
		if field.IsDynamic {
			hasDynamicSchema = true
		}
		if field.IsPartitionKey {
			hasPartitionKey = true
		}
		if field.DataType == entity.FieldTypeFloatVector ||
			field.DataType == entity.FieldTypeBinaryVector ||
			field.DataType == entity.FieldTypeBFloat16Vector ||
			field.DataType == entity.FieldTypeFloat16Vector ||
			field.DataType == entity.FieldTypeSparseVector {
			vectors++
		}
	}
	if vectors <= 0 {
		return errors.New("vector field not set")
	}
	switch {
	case hasJSON && c.config.hasFlags(disableJSON):
		return ErrFeatureNotSupported
	case hasDynamicSchema && c.config.hasFlags(disableDynamicSchema):
		return ErrFeatureNotSupported
	case hasPartitionKey && c.config.hasFlags(disableParitionKey):
		return ErrFeatureNotSupported
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
	req := &milvuspb.DescribeCollectionRequest{
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
		Schema:           entity.NewSchema().ReadProto(resp.GetSchema()),
		PhysicalChannels: resp.GetPhysicalChannelNames(),
		VirtualChannels:  resp.GetVirtualChannelNames(),
		ConsistencyLevel: entity.ConsistencyLevel(resp.ConsistencyLevel),
		ShardNum:         resp.GetShardsNum(),
		Properties:       entity.KvPairsMap(resp.GetProperties()),
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
func (c *GrpcClient) DropCollection(ctx context.Context, collName string, opts ...DropCollectionOption) error {
	if c.Service == nil {
		return ErrClientNotReady
	}
	if err := c.checkCollectionExists(ctx, collName); err != nil {
		return err
	}

	req := &milvuspb.DropCollectionRequest{
		CollectionName: collName,
	}
	for _, opt := range opts {
		opt(req)
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

	req := &milvuspb.HasCollectionRequest{
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

	req := &milvuspb.GetCollectionStatisticsRequest{
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

	req := &milvuspb.ShowCollectionsRequest{
		Type:            milvuspb.ShowType_InMemory,
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

// RenameCollection performs renaming for provided collection.
func (c *GrpcClient) RenameCollection(ctx context.Context, collName, newName string) error {
	if c.Service == nil {
		return ErrClientNotReady
	}

	if err := c.checkCollectionExists(ctx, collName); err != nil {
		return err
	}

	req := &milvuspb.RenameCollectionRequest{
		OldName: collName,
		NewName: newName,
	}
	resp, err := c.Service.RenameCollection(ctx, req)
	if err != nil {
		return err
	}
	return handleRespStatus(resp)
}

// LoadCollection load collection into memory
func (c *GrpcClient) LoadCollection(ctx context.Context, collName string, async bool, opts ...LoadCollectionOption) error {
	if c.Service == nil {
		return ErrClientNotReady
	}

	if err := c.checkCollectionExists(ctx, collName); err != nil {
		return err
	}

	req := &milvuspb.LoadCollectionRequest{
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
		ticker := time.NewTicker(200 * time.Millisecond)
		defer ticker.Stop()
		for {
			select {
			case <-ctx.Done():
				return ctx.Err()
			case <-ticker.C:
				progress, err := c.getLoadingProgress(ctx, collName)
				if err != nil {
					return err
				}
				if progress == 100 {
					return nil
				}
			}
		}
	}
	return nil
}

// ReleaseCollection release loaded collection
func (c *GrpcClient) ReleaseCollection(ctx context.Context, collName string, opts ...ReleaseCollectionOption) error {
	if c.Service == nil {
		return ErrClientNotReady
	}
	if err := c.checkCollectionExists(ctx, collName); err != nil {
		return err
	}

	req := &milvuspb.ReleaseCollectionRequest{
		DbName:         "", // reserved
		CollectionName: collName,
	}
	for _, opt := range opts {
		opt(req)
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

	req := &milvuspb.GetReplicasRequest{
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

	req := &milvuspb.GetLoadingProgressRequest{
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

	req := &milvuspb.GetLoadStateRequest{
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

	props := make([]*commonpb.KeyValuePair, 0, len(attrs))
	for _, attr := range attrs {
		k, v := attr.KeyValue()
		if _, exists := keys[k]; exists {
			return errors.New("duplicated attributed received")
		}
		keys[k] = struct{}{}
		props = append(props, &commonpb.KeyValuePair{
			Key:   k,
			Value: v,
		})
	}

	req := &milvuspb.AlterCollectionRequest{
		CollectionName: collName,
		Properties:     props,
	}

	resp, err := c.Service.AlterCollection(ctx, req)
	if err != nil {
		return err
	}
	return handleRespStatus(resp)
}

func (c *GrpcClient) getLoadingProgress(ctx context.Context, collectionName string, partitionNames ...string) (int64, error) {
	req := &milvuspb.GetLoadingProgressRequest{
		Base:           &commonpb.MsgBase{},
		DbName:         "",
		CollectionName: collectionName,
		PartitionNames: partitionNames,
	}

	resp, err := c.Service.GetLoadingProgress(ctx, req)
	if err != nil {
		return -1, err
	}
	if err := handleRespStatus(resp.GetStatus()); err != nil {
		return -1, err
	}
	return resp.GetProgress(), nil
}
