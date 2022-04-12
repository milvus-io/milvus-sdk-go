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
	"errors"
	"fmt"
	"time"

	"github.com/golang/protobuf/proto"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
	"github.com/milvus-io/milvus-sdk-go/v2/internal/proto/common"
	"github.com/milvus-io/milvus-sdk-go/v2/internal/proto/server"
	"google.golang.org/grpc"
)

// grpcClient, uses default grpc service definition to connect with Milvus2.0
type grpcClient struct {
	conn    *grpc.ClientConn           // grpc connection instance
	service server.MilvusServiceClient // service client stub
}

// connect connect to service
func (c *grpcClient) connect(ctx context.Context, addr string, opts ...grpc.DialOption) error {

	// if not options provided, use default settings
	if len(opts) == 0 {
		opts = append(opts, grpc.WithInsecure(),
			grpc.WithBlock(),                //block connect until healthy or timeout
			grpc.WithTimeout(2*time.Second)) // set connect timeout to 2 Second
	}

	conn, err := grpc.DialContext(ctx, addr, opts...)
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
		err := c.conn.Close()
		c.conn = nil
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
		if len(resp.GetInMemoryPercentages()) > idx {
			collection.Loaded = resp.GetInMemoryPercentages()[idx] == 100
		}
		collections = append(collections, collection)
	}
	return collections, nil
}

// CreateCollection create collection with specified schema
func (c *grpcClient) CreateCollection(ctx context.Context, collSchema *entity.Schema, shardNum int32) error {
	if c.service == nil {
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
	}
	resp, err := c.service.CreateCollection(ctx, req)
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
			if field.DataType != entity.FieldTypeInt64 { // string key not supported yet
				return errors.New("only int64 column can be primary key for now")
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

func (c *grpcClient) checkCollectionExists(ctx context.Context, collName string) error {
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
func (c *grpcClient) DescribeCollection(ctx context.Context, collName string) (*entity.Collection, error) {
	if c.service == nil {
		return nil, ErrClientNotReady
	}

	if err := c.checkCollectionExists(ctx, collName); err != nil {
		return nil, err
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
	if err := c.checkCollectionExists(ctx, collName); err != nil {
		return err
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
	if err := handleRespStatus(resp.GetStatus()); err != nil {
		return false, err
	}
	return resp.GetValue(), nil
}

// GetCollectionStatistcis show collection statistics
func (c *grpcClient) GetCollectionStatistics(ctx context.Context, collName string) (map[string]string, error) {
	if c.service == nil {
		return nil, ErrClientNotReady
	}

	if err := c.checkCollectionExists(ctx, collName); err != nil {
		return nil, err
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

// ShowCollection show collection status, used to check whether it is loaded or not
func (c *grpcClient) ShowCollection(ctx context.Context, collName string) (*entity.Collection, error) {
	if c.service == nil {
		return nil, ErrClientNotReady
	}
	if err := c.checkCollectionExists(ctx, collName); err != nil {
		return nil, err
	}

	req := &server.ShowCollectionsRequest{
		Type:            server.ShowType_InMemory,
		CollectionNames: []string{collName},
	}

	resp, err := c.service.ShowCollections(ctx, req)
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
		Loaded: resp.InMemoryPercentages[0] == 100, // TODO silverxia, the percentage can be either 0 or 100
	}, nil
}

// LoadCollection load collection into memory
func (c *grpcClient) LoadCollection(ctx context.Context, collName string, async bool) error {
	if c.service == nil {
		return ErrClientNotReady
	}

	if err := c.checkCollectionExists(ctx, collName); err != nil {
		return err
	}

	req := &server.LoadCollectionRequest{
		CollectionName: collName,
	}
	resp, err := c.service.LoadCollection(ctx, req)
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
func (c *grpcClient) ReleaseCollection(ctx context.Context, collName string) error {
	if c.service == nil {
		return ErrClientNotReady
	}
	if err := c.checkCollectionExists(ctx, collName); err != nil {
		return err
	}

	req := &server.ReleaseCollectionRequest{
		DbName:         "", //reserved
		CollectionName: collName,
	}
	resp, err := c.service.ReleaseCollection(ctx, req)
	if err != nil {
		return err
	}
	return handleRespStatus(resp)
}
