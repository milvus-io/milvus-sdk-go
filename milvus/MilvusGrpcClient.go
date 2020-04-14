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
	"time"

	pb "github.com/milvus-io/milvus-sdk-go/milvus/grpc/gen"
)

var timeout time.Duration = 10 * time.Second

// MilvusGrpcClient call grpc generated code interface
type MilvusGrpcClient interface {
	CreateCollection(collectionSchema pb.CollectionSchema) (pb.Status, error)

	HasCollection(collectionName pb.CollectionName) (pb.BoolReply, error)

	DescribeCollection(collectionName pb.CollectionName) (pb.CollectionSchema, error)

	CountCollection(collectionName pb.CollectionName) (pb.CollectionRowCount, error)

	ShowCollections() (pb.CollectionNameList, error)

	ShowCollectionInfos(collectionName pb.CollectionName) (pb.CollectionInfo, error)

	DropCollection(collectionName pb.CollectionName) (pb.Status, error)

	CreateIndex(indexParam pb.IndexParam) (pb.Status, error)

	DescribeIndex(collectionName pb.CollectionName) (pb.IndexParam, error)

	DropIndex(collectionName pb.CollectionName) (pb.Status, error)

	CreatePartition(partitionParam pb.PartitionParam) (pb.Status, error)

	ShowPartitions(collectionName pb.CollectionName) (pb.PartitionList, error)

	DropPartition(partitionParam pb.PartitionParam) (pb.Status, error)

	Insert(insertParam pb.InsertParam) (pb.VectorIds, error)

	GetVectorByID(identity pb.VectorIdentity) (pb.VectorData, error)

	GetVectorIDs(param pb.GetVectorIDsParam) (pb.VectorIds, error)

	Search(searchParam pb.SearchParam) (*pb.TopKQueryResult, error)

	SearchInFiles(searchInFilesParam pb.SearchInFilesParam) (*pb.TopKQueryResult, error)

	Cmd(command pb.Command) (pb.StringReply, error)

	DeleteByID(param pb.DeleteByIDParam) (pb.Status, error)

	PreloadCollection(collectionName pb.CollectionName) (pb.Status, error)

	Flush(param pb.FlushParam) (pb.Status, error)

	Compact(name pb.CollectionName) (pb.Status, error)
}

type milvusGrpcClient struct {
	serviceInstance pb.MilvusServiceClient
}

// NewMilvusGrpcClient is the constructor of MilvusGrpcClient
func NewMilvusGrpcClient(client pb.MilvusServiceClient) MilvusGrpcClient {
	return &milvusGrpcClient{client}
}

func (grpcClient *milvusGrpcClient) CreateCollection(collectionSchema pb.CollectionSchema) (pb.Status, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	reply, err := grpcClient.serviceInstance.CreateCollection(ctx, &collectionSchema)
	if err != nil {
		return pb.Status{0, "", struct{}{}, nil, 0,}, err
	}
	return *reply, err
}

func (grpcClient *milvusGrpcClient) HasCollection(collectionName pb.CollectionName) (pb.BoolReply, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	boolReply, err := grpcClient.serviceInstance.HasCollection(ctx, &collectionName)
	if err != nil {
		return pb.BoolReply{nil, false, struct{}{}, nil, 0,}, err
	}
	return *boolReply, err
}

func (grpcClient *milvusGrpcClient) DescribeCollection(collectionName pb.CollectionName) (pb.CollectionSchema, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	collectionSchema, err := grpcClient.serviceInstance.DescribeCollection(ctx, &collectionName)
	if err != nil {
		return pb.CollectionSchema{nil, "", 0, 0, 0, nil, struct{}{}, nil, 0,}, err
	}
	return *collectionSchema, err
}

func (grpcClient *milvusGrpcClient) CountCollection(collectionName pb.CollectionName) (pb.CollectionRowCount, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	count, err := grpcClient.serviceInstance.CountCollection(ctx, &collectionName)
	if err != nil {
		return pb.CollectionRowCount{nil, 0, struct{}{}, nil, 0,}, err
	}
	return *count, err
}

func (grpcClient *milvusGrpcClient) ShowCollections() (pb.CollectionNameList, error) {
	cmd := pb.Command{"", struct{}{}, nil, 0}
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	collectionNameList, err := grpcClient.serviceInstance.ShowCollections(ctx, &cmd)
	if err != nil {
		return pb.CollectionNameList{nil, nil, struct{}{}, nil, 0,}, err
	}
	return *collectionNameList, err
}

func (grpcClient *milvusGrpcClient) ShowCollectionInfos(collectionName pb.CollectionName) (pb.CollectionInfo, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	collectionInfo, err := grpcClient.serviceInstance.ShowCollectionInfo(ctx, &collectionName)
	if err != nil {
		return pb.CollectionInfo{nil, 0, nil, struct{}{}, nil, 0,}, err
	}
	return *collectionInfo, err
}

func (grpcClient *milvusGrpcClient) DropCollection(collectionName pb.CollectionName) (pb.Status, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	status, err := grpcClient.serviceInstance.DropCollection(ctx, &collectionName)
	if err != nil {
		return pb.Status{0, "", struct{}{}, nil, 0,}, err
	}
	return *status, err
}

func (grpcClient *milvusGrpcClient) CreateIndex(indexParam pb.IndexParam) (pb.Status, error) {
	ctx := context.Background()
	status, err := grpcClient.serviceInstance.CreateIndex(ctx, &indexParam)
	if err != nil {
		return pb.Status{0, "", struct{}{}, nil, 0,}, err
	}
	return *status, err
}

func (grpcClient *milvusGrpcClient) DescribeIndex(collectionName pb.CollectionName) (pb.IndexParam, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	indexParam, err := grpcClient.serviceInstance.DescribeIndex(ctx, &collectionName)
	if err != nil {
		return pb.IndexParam{nil, "", 0, nil, struct{}{}, nil, 0,}, err
	}
	return *indexParam, err
}

func (grpcClient *milvusGrpcClient) DropIndex(collectionName pb.CollectionName) (pb.Status, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	status, err := grpcClient.serviceInstance.DropIndex(ctx, &collectionName)
	if err != nil {
		return pb.Status{0, "", struct{}{}, nil, 0,}, err
	}
	return *status, err
}

func (grpcClient *milvusGrpcClient) CreatePartition(partitionParam pb.PartitionParam) (pb.Status, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	status, err := grpcClient.serviceInstance.CreatePartition(ctx, &partitionParam)
	if err != nil {
		return pb.Status{0, "", struct{}{}, nil, 0,}, err
	}
	return *status, err
}

func (grpcClient *milvusGrpcClient) ShowPartitions(collectionName pb.CollectionName) (pb.PartitionList, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	status, err := grpcClient.serviceInstance.ShowPartitions(ctx, &collectionName)
	if err != nil {
		return pb.PartitionList{nil, nil, struct{}{}, nil, 0,}, err
	}
	return *status, err
}

func (grpcClient *milvusGrpcClient) DropPartition(partitionParam pb.PartitionParam) (pb.Status, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	status, err := grpcClient.serviceInstance.DropPartition(ctx, &partitionParam)
	if err != nil {
		return pb.Status{0, "", struct{}{}, nil, 0,}, err
	}
	return *status, err
}

func (grpcClient *milvusGrpcClient) Insert(insertParam pb.InsertParam) (pb.VectorIds, error) {
	ctx := context.Background()
	vectorIds, err := grpcClient.serviceInstance.Insert(ctx, &insertParam)
	if err != nil {
		return pb.VectorIds{nil, nil, struct{}{}, nil, 0,}, err
	}
	return *vectorIds, err
}

func (grpcClient *milvusGrpcClient) GetVectorByID(identity pb.VectorIdentity) (pb.VectorData, error) {
	ctx := context.Background()
	status, err := grpcClient.serviceInstance.GetVectorByID(ctx, &identity)
	if err != nil {
		return pb.VectorData{nil, nil, struct{}{}, nil, 0,}, err
	}
	return *status, err
}

func (grpcClient *milvusGrpcClient) GetVectorIDs(param pb.GetVectorIDsParam) (pb.VectorIds, error) {
	ctx := context.Background()
	status, err := grpcClient.serviceInstance.GetVectorIDs(ctx, &param)
	if err != nil {
		return pb.VectorIds{nil, nil, struct{}{}, nil, 0,}, err
	}
	return *status, err
}

func (grpcClient *milvusGrpcClient) Search(searchParam pb.SearchParam) (*pb.TopKQueryResult, error) {
	ctx := context.Background()
	topkQueryResult, err := grpcClient.serviceInstance.Search(ctx, &searchParam)
	if err != nil {
		return &pb.TopKQueryResult{nil, 0, nil, nil, struct{}{}, nil, 0,}, err
	}
	return topkQueryResult, err
}

func (grpcClient *milvusGrpcClient) SearchInFiles(searchInFilesParam pb.SearchInFilesParam) (*pb.TopKQueryResult, error) {
	ctx := context.Background()
	topkQueryResult, err := grpcClient.serviceInstance.SearchInFiles(ctx, &searchInFilesParam)
	if err != nil {
		return &pb.TopKQueryResult{nil, 0, nil, nil, struct{}{}, nil, 0,}, err
	}
	return topkQueryResult, err
}

func (grpcClient *milvusGrpcClient) Cmd(command pb.Command) (pb.StringReply, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	stringReply, err := grpcClient.serviceInstance.Cmd(ctx, &command)
	if err != nil {
		return pb.StringReply{nil, "", struct{}{}, nil, 0,}, err
	}
	return *stringReply, err
}

func (grpcClient *milvusGrpcClient) DeleteByID(param pb.DeleteByIDParam) (pb.Status, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	status, err := grpcClient.serviceInstance.DeleteByID(ctx, &param)
	if err != nil {
		return pb.Status{0, "", struct{}{}, nil, 0,}, err
	}
	return *status, err
}

func (grpcClient *milvusGrpcClient) PreloadCollection(collectionName pb.CollectionName) (pb.Status, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	status, err := grpcClient.serviceInstance.PreloadCollection(ctx, &collectionName)
	if err != nil {
		return pb.Status{0, "", struct{}{}, nil, 0,}, err
	}
	return *status, err
}

func (grpcClient *milvusGrpcClient) Flush(param pb.FlushParam) (pb.Status, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	status, err := grpcClient.serviceInstance.Flush(ctx, &param)
	if err != nil {
		return pb.Status{0, "", struct{}{}, nil, 0,}, err
	}
	return *status, err
}

func (grpcClient *milvusGrpcClient) Compact(collectionName pb.CollectionName) (pb.Status, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	status, err := grpcClient.serviceInstance.Compact(ctx, &collectionName)
	if err != nil {
		return pb.Status{0, "", struct{}{}, nil, 0,}, err
	}
	return *status, err
}
