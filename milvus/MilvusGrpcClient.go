/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

// package milvus
package milvus

import (
	"context"

	pb "github.com/milvus-io/milvus-sdk-go/milvus/grpc/gen"
)

// MilvusGrpcClient call grpc generated code interface
type MilvusGrpcClient interface {
	CreateCollection(ctx context.Context, mapping pb.Mapping) (pb.Status, error)

	HasCollection(ctx context.Context, collectionName pb.CollectionName) (pb.BoolReply, error)

	DescribeCollection(ctx context.Context, collectionName pb.CollectionName) (pb.Mapping, error)

	CountCollection(ctx context.Context, collectionName pb.CollectionName) (pb.CollectionRowCount, error)

	ShowCollections(ctx context.Context) (pb.CollectionNameList, error)

	ShowCollectionInfo(ctx context.Context, collectionName pb.CollectionName) (pb.CollectionInfo, error)

	DropCollection(ctx context.Context, collectionName pb.CollectionName) (pb.Status, error)

	CreateIndex(ctx context.Context, indexParam pb.IndexParam) (pb.Status, error)

	DropIndex(ctx context.Context, indexParam pb.IndexParam) (pb.Status, error)

	CreatePartition(ctx context.Context, partitionParam pb.PartitionParam) (pb.Status, error)

	ShowPartitions(ctx context.Context, collectionName pb.CollectionName) (pb.PartitionList, error)

	HasPartition(ctx context.Context, param pb.PartitionParam) (pb.BoolReply, error)

	DropPartition(ctx context.Context, partitionParam pb.PartitionParam) (pb.Status, error)

	Insert(ctx context.Context, insertParam pb.InsertParam) (pb.EntityIds, error)

	GetEntityByID(ctx context.Context, identity pb.EntityIdentity) (pb.Entities, error)

	GetEntityIDs(ctx context.Context, param pb.GetEntityIDsParam) (pb.EntityIds, error)

	Search(ctx context.Context, searchParam pb.SearchParam) (*pb.QueryResult, error)

	Cmd(ctx context.Context, command pb.Command) (pb.StringReply, error)

	DeleteByID(ctx context.Context, param pb.DeleteByIDParam) (pb.Status, error)

	PreloadCollection(ctx context.Context, collectionName pb.CollectionName) (pb.Status, error)

	Flush(ctx context.Context, param pb.FlushParam) (pb.Status, error)

	Compact(ctx context.Context, compactParam pb.CompactParam) (pb.Status, error)
}

type milvusGrpcClient struct {
	serviceInstance pb.MilvusServiceClient
}

// NewMilvusGrpcClient is the constructor of MilvusGrpcClient
func NewMilvusGrpcClient(client pb.MilvusServiceClient) MilvusGrpcClient {
	return &milvusGrpcClient{client}
}

func (grpcClient *milvusGrpcClient) CreateCollection(ctx context.Context, mapping pb.Mapping) (pb.Status, error) {
	reply, err := grpcClient.serviceInstance.CreateCollection(ctx, &mapping)
	if err != nil {
		return pb.Status{0, "", struct{}{}, nil, 0}, err
	}
	return *reply, err
}

func (grpcClient *milvusGrpcClient) HasCollection(ctx context.Context, collectionName pb.CollectionName) (pb.BoolReply, error) {
	boolReply, err := grpcClient.serviceInstance.HasCollection(ctx, &collectionName)
	if err != nil {
		return pb.BoolReply{nil, false, struct{}{}, nil, 0}, err
	}
	return *boolReply, err
}

func (grpcClient *milvusGrpcClient) DescribeCollection(ctx context.Context, collectionName pb.CollectionName) (pb.Mapping, error) {
	collectionSchema, err := grpcClient.serviceInstance.DescribeCollection(ctx, &collectionName)
	if err != nil {
		return pb.Mapping{nil, "", nil, nil, struct{}{}, nil, 0}, err
	}
	return *collectionSchema, err
}

func (grpcClient *milvusGrpcClient) CountCollection(ctx context.Context, collectionName pb.CollectionName) (pb.CollectionRowCount, error) {
	count, err := grpcClient.serviceInstance.CountCollection(ctx, &collectionName)
	if err != nil {
		return pb.CollectionRowCount{nil, 0, struct{}{}, nil, 0}, err
	}
	return *count, err
}

func (grpcClient *milvusGrpcClient) ShowCollections(ctx context.Context) (pb.CollectionNameList, error) {
	cmd := pb.Command{"", struct{}{}, nil, 0}
	collectionNameList, err := grpcClient.serviceInstance.ShowCollections(ctx, &cmd)
	if err != nil {
		return pb.CollectionNameList{nil, nil, struct{}{}, nil, 0}, err
	}
	return *collectionNameList, err
}

func (grpcClient *milvusGrpcClient) ShowCollectionInfo(ctx context.Context, collectionName pb.CollectionName) (pb.CollectionInfo, error) {
	collectionInfo, err := grpcClient.serviceInstance.ShowCollectionInfo(ctx, &collectionName)
	if err != nil {
		return pb.CollectionInfo{nil, "", struct{}{}, nil, 0}, err
	}
	return *collectionInfo, err
}

func (grpcClient *milvusGrpcClient) DropCollection(ctx context.Context, collectionName pb.CollectionName) (pb.Status, error) {
	status, err := grpcClient.serviceInstance.DropCollection(ctx, &collectionName)
	if err != nil {
		return pb.Status{0, "", struct{}{}, nil, 0}, err
	}
	return *status, err
}

func (grpcClient *milvusGrpcClient) CreateIndex(ctx context.Context, indexParam pb.IndexParam) (pb.Status, error) {
	status, err := grpcClient.serviceInstance.CreateIndex(ctx, &indexParam)
	if err != nil {
		return pb.Status{0, "", struct{}{}, nil, 0}, err
	}
	return *status, err
}

func (grpcClient *milvusGrpcClient) DropIndex(ctx context.Context, indexParam pb.IndexParam) (pb.Status, error) {
	status, err := grpcClient.serviceInstance.DropIndex(ctx, &indexParam)
	if err != nil {
		return pb.Status{0, "", struct{}{}, nil, 0}, err
	}
	return *status, err
}

func (grpcClient *milvusGrpcClient) CreatePartition(ctx context.Context, partitionParam pb.PartitionParam) (pb.Status, error) {
	status, err := grpcClient.serviceInstance.CreatePartition(ctx, &partitionParam)
	if err != nil {
		return pb.Status{0, "", struct{}{}, nil, 0}, err
	}
	return *status, err
}

func (grpcClient *milvusGrpcClient) ShowPartitions(ctx context.Context, collectionName pb.CollectionName) (pb.PartitionList, error) {
	status, err := grpcClient.serviceInstance.ShowPartitions(ctx, &collectionName)
	if err != nil {
		return pb.PartitionList{nil, nil, struct{}{}, nil, 0}, err
	}
	return *status, err
}

func (grpcClient *milvusGrpcClient) HasPartition(ctx context.Context, param pb.PartitionParam) (pb.BoolReply, error) {
	reply, err := grpcClient.serviceInstance.HasPartition(ctx, &param)
	if err != nil {
		return pb.BoolReply{nil, false, struct{}{}, nil, 0}, err
	}
	return *reply, err
}

func (grpcClient *milvusGrpcClient) DropPartition(ctx context.Context, partitionParam pb.PartitionParam) (pb.Status, error) {
	status, err := grpcClient.serviceInstance.DropPartition(ctx, &partitionParam)
	if err != nil {
		return pb.Status{0, "", struct{}{}, nil, 0}, err
	}
	return *status, err
}

func (grpcClient *milvusGrpcClient) Insert(ctx context.Context, insertParam pb.InsertParam) (pb.EntityIds, error) {
	entityIds, err := grpcClient.serviceInstance.Insert(ctx, &insertParam)
	if err != nil {
		return pb.EntityIds{nil, nil, struct{}{}, nil, 0}, err
	}
	return *entityIds, err
}

func (grpcClient *milvusGrpcClient) GetEntityByID(ctx context.Context, identity pb.EntityIdentity) (pb.Entities, error) {
	vectorsData, err := grpcClient.serviceInstance.GetEntityByID(ctx, &identity)
	if err != nil {
		return pb.Entities{nil, nil, nil, nil, struct{}{}, nil, 0}, err
	}
	return *vectorsData, err
}

func (grpcClient *milvusGrpcClient) GetEntityIDs(ctx context.Context, param pb.GetEntityIDsParam) (pb.EntityIds, error) {
	status, err := grpcClient.serviceInstance.GetEntityIDs(ctx, &param)
	if err != nil {
		return pb.EntityIds{nil, nil, struct{}{}, nil, 0}, err
	}
	return *status, err
}

func (grpcClient *milvusGrpcClient) Search(ctx context.Context, searchParam pb.SearchParam) (*pb.QueryResult, error) {
	queryResult, err := grpcClient.serviceInstance.Search(ctx, &searchParam)
	if err != nil {
		return &pb.QueryResult{nil, nil, 0, nil, nil, nil, struct{}{}, nil, 0}, err
	}
	return queryResult, err
}

func (grpcClient *milvusGrpcClient) Cmd(ctx context.Context, command pb.Command) (pb.StringReply, error) {
	stringReply, err := grpcClient.serviceInstance.Cmd(ctx, &command)
	if err != nil {
		return pb.StringReply{nil, "", struct{}{}, nil, 0}, err
	}
	return *stringReply, err
}

func (grpcClient *milvusGrpcClient) DeleteByID(ctx context.Context, param pb.DeleteByIDParam) (pb.Status, error) {
	status, err := grpcClient.serviceInstance.DeleteByID(ctx, &param)
	if err != nil {
		return pb.Status{0, "", struct{}{}, nil, 0}, err
	}
	return *status, err
}

func (grpcClient *milvusGrpcClient) PreloadCollection(ctx context.Context, collectionName pb.CollectionName) (pb.Status, error) {
	status, err := grpcClient.serviceInstance.PreloadCollection(ctx, &collectionName)
	if err != nil {
		return pb.Status{0, "", struct{}{}, nil, 0}, err
	}
	return *status, err
}

func (grpcClient *milvusGrpcClient) Flush(ctx context.Context, param pb.FlushParam) (pb.Status, error) {
	status, err := grpcClient.serviceInstance.Flush(ctx, &param)
	if err != nil {
		return pb.Status{0, "", struct{}{}, nil, 0}, err
	}
	return *status, err
}

func (grpcClient *milvusGrpcClient) Compact(ctx context.Context, compactParam pb.CompactParam) (pb.Status, error) {
	status, err := grpcClient.serviceInstance.Compact(ctx, &compactParam)
	if err != nil {
		return pb.Status{0, "", struct{}{}, nil, 0}, err
	}
	return *status, err
}
