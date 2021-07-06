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

// package client provides milvus client functions
package client

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/golang/protobuf/proto"
	"github.com/milvus-io/milvus-sdk-go/entity"
	"github.com/milvus-io/milvus-sdk-go/internal/proto/common"
	"github.com/milvus-io/milvus-sdk-go/internal/proto/server"
	"google.golang.org/grpc"
)

var (
	//ErrClientNotReady error indicates client not ready
	ErrClientNotReady = errors.New("client not ready")
	//ErrStatusNil error indicates response has nil status
	ErrStatusNil = errors.New("response status is nil")
)

type ErrServiceFailed error

// grpcClient, uses default grpc service definition to connect with Milvus2.0
type grpcClient struct {
	conn    *grpc.ClientConn           // grpc connection instance
	service server.MilvusServiceClient // service client stub
}

// Connect connect to service
func (c *grpcClient) Connect(ctx context.Context, addr string) error {
	conn, err := grpc.DialContext(ctx, addr,
		grpc.WithInsecure(),
		grpc.WithBlock(),                //block connect until healthy or timeout
		grpc.WithTimeout(2*time.Second), // set connect timeout to 2 Second
	)
	if err != nil {
		return err
	}
	c.conn = conn
	c.service = server.NewMilvusServiceClient(c.conn)
	return nil
}

// Close close the connection
func (c *grpcClient) Close() error {
	if c.conn != nil {
		return c.conn.Close()
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
		return ErrServiceFailed(errors.New("service failed"))
	}
	return nil
}

// ListCollections list collections from connection
// Note that schema info are not provided in collection list
func (c *grpcClient) ListCollections(ctx context.Context) ([]*entity.Collection, error) {
	if c.service == nil {
		return []*entity.Collection{}, ErrClientNotReady
	}
	req := &server.ShowCollectionsRequest{
		DbName:    "",
		TimeStamp: 0, // means now
	}
	resp, err := c.service.ShowCollections(ctx, req)
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
		collections = append(collections, collection)
	}
	return collections, nil
}

// CreateCollection create collection with specified schema
func (c *grpcClient) CreateCollection(ctx context.Context, collSchema entity.Schema, shardNum int32) error {
	if c.service == nil {
		return ErrClientNotReady
	}
	sch := collSchema.ProtoMessage()
	bs, err := proto.Marshal(sch)
	if err != nil {
		fmt.Println(err.Error())
	}
	req := &server.CreateCollectionRequest{
		DbName:         "", // reserved fields, not used for now
		CollectionName: collSchema.CollectionName,
		Schema:         bs,
		ShardsNum:      shardNum,
	}
	resp, err := c.service.CreateCollection(ctx, req)
	if err != nil {
		return err
	}
	err = handleRespStatus(resp)
	if err != nil {
		return nil
	}
	return nil
}

// DescribeCollection describe the collection by name
func (c *grpcClient) DescribeCollection(ctx context.Context, collName string) (*entity.Collection, error) {
	if c.service == nil {
		return nil, ErrClientNotReady
	}

	req := &server.DescribeCollectionRequest{
		CollectionName: collName,
	}
	resp, err := c.service.DescribeCollection(ctx, req)
	if err != nil {
		return nil, err
	}
	err = handleRespStatus(resp.GetStatus())
	if err != nil {
		return nil, err
	}
	collection := &entity.Collection{
		ID:               resp.GetCollectionID(),
		Schema:           (&entity.Schema{}).ReadProto(resp.GetSchema()),
		PhysicalChannels: resp.GetPhysicalChannelNames(),
		VirtualChannels:  resp.GetVirtualChannelNames(),
	}
	collection.Name = collection.Schema.CollectionName
	return collection, nil
}

// DropCollection drop collection by name
func (c *grpcClient) DropCollection(ctx context.Context, collName string) error {
	if c.service == nil {
		return ErrClientNotReady
	}

	req := &server.DropCollectionRequest{
		CollectionName: collName,
	}
	resp, err := c.service.DropCollection(ctx, req)
	if err != nil {
		return err
	}
	return handleRespStatus(resp)
}

// HasCollection check whether collection name exists
func (c *grpcClient) HasCollection(ctx context.Context, collName string) (bool, error) {
	if c.service == nil {
		return false, ErrClientNotReady
	}

	req := &server.HasCollectionRequest{
		DbName:         "", // reserved
		CollectionName: collName,
		TimeStamp:      0, // 0 for now
	}

	resp, err := c.service.HasCollection(ctx, req)
	if err != nil {
		return false, err
	}
	if resp.Status.ErrorCode != common.ErrorCode_Success {
		return false, errors.New("request failed")
	}
	return resp.GetValue(), nil
}

// GetCollectionStatistcis show collection statistics
func (c *grpcClient) GetCollectionStatistics(ctx context.Context, collName string) (map[string]string, error) {
	if c.service == nil {
		return nil, ErrClientNotReady
	}

	req := &server.GetCollectionStatisticsRequest{
		CollectionName: collName,
	}
	resp, err := c.service.GetCollectionStatistics(ctx, req)
	if err != nil {
		return nil, err
	}
	if err := handleRespStatus(resp.GetStatus()); err != nil {
		return nil, err
	}
	return entity.KvPairsMap(resp.GetStats()), nil
}

// LoadCollection load collection into memory
func (c *grpcClient) LoadCollection(ctx context.Context, collName string) error {
	if c.service == nil {
		return ErrClientNotReady
	}
	req := &server.LoadCollectionRequest{
		CollectionName: collName,
	}
	resp, err := c.service.LoadCollection(ctx, req)
	if err != nil {
		return err
	}
	return handleRespStatus(resp)
}

// CreatePartition create partition for collection
func (c *grpcClient) CreatePartition(ctx context.Context, collName string, partitionName string) error {
	if c.service == nil {
		return ErrClientNotReady
	}
	req := &server.CreatePartitionRequest{
		DbName:         "", //reserved
		CollectionName: collName,
		PartitionName:  partitionName,
	}
	resp, err := c.service.CreatePartition(ctx, req)
	if err != nil {
		return err
	}
	return handleRespStatus(resp)
}

// DropPartition drop partition from collection
func (c *grpcClient) DropPartition(ctx context.Context, collName string, partitionName string) error {
	if c.service == nil {
		return ErrClientNotReady
	}
	req := &server.DropPartitionRequest{
		DbName:         "",
		CollectionName: collName,
		PartitionName:  partitionName,
	}
	resp, err := c.service.DropPartition(ctx, req)
	if err != nil {
		return err
	}
	return handleRespStatus(resp)
}

// HasPartition check whether specified partition exists
func (c *grpcClient) HasPartition(ctx context.Context, collName string, partitionName string) (bool, error) {
	if c.service == nil {
		return false, ErrClientNotReady
	}
	req := &server.HasPartitionRequest{
		DbName:         "", // reserved
		CollectionName: collName,
		PartitionName:  partitionName,
	}
	resp, err := c.service.HasPartition(ctx, req)
	if err != nil {
		return false, err
	}
	if resp.GetStatus().GetErrorCode() != common.ErrorCode_Success {
		return false, errors.New("request failed")
	}
	return resp.GetValue(), nil
}

// ShowPartitions list all paritions from collection
func (c *grpcClient) ShowPartitions(ctx context.Context, collName string) ([]*entity.Partition, error) {
	if c.service == nil {
		return []*entity.Partition{}, ErrClientNotReady
	}
	req := &server.ShowPartitionsRequest{
		DbName:         "", // reserved
		CollectionName: collName,
	}
	resp, err := c.service.ShowPartitions(ctx, req)
	if err != nil {
		return []*entity.Partition{}, err
	}
	partitions := make([]*entity.Partition, 0, len(resp.GetPartitionIDs()))
	for idx, partitionID := range resp.GetPartitionIDs() {
		partitions = append(partitions, &entity.Partition{ID: partitionID, Name: resp.GetPartitionNames()[idx]})
	}
	return partitions, nil
}

func (c *grpcClient) Insert(ctx context.Context, collName string, partitionName string, columns []entity.Column) error {
	if c.service == nil {
		return ErrClientNotReady
	}
	req := &server.InsertRequest{
		DbName:         "", // reserved
		CollectionName: collName,
		PartitionName:  partitionName,
	}
	if req.PartitionName == "" {
		req.PartitionName = "_default" // use default partition
	}
	for _, column := range columns {
		req.FieldsData = append(req.FieldsData, column.FieldData())
	}
	resp, err := c.service.Insert(ctx, req)
	if err != nil {
		return err
	}
	return handleRespStatus(resp.GetStatus())
}
