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
	CreateCollection(ctx context.Context, collectionSchema pb.CollectionSchema) (pb.Status, error)

	HasCollection(ctx context.Context, collectionName pb.CollectionName) (pb.BoolReply, error)

	DescribeCollection(ctx context.Context, collectionName pb.CollectionName) (pb.CollectionSchema, error)

	CountCollection(ctx context.Context, collectionName pb.CollectionName) (pb.CollectionRowCount, error)

	ShowCollections(ctx context.Context) (pb.CollectionNameList, error)

	ShowCollectionInfo(ctx context.Context, collectionName pb.CollectionName) (pb.CollectionInfo, error)

	DropCollection(ctx context.Context, collectionName pb.CollectionName) (pb.Status, error)

	CreateIndex(ctx context.Context, indexParam pb.IndexParam) (pb.Status, error)

	DescribeIndex(ctx context.Context, collectionName pb.CollectionName) (pb.IndexParam, error)

	DropIndex(ctx context.Context, collectionName pb.CollectionName) (pb.Status, error)

	CreatePartition(ctx context.Context, partitionParam pb.PartitionParam) (pb.Status, error)

	ShowPartitions(ctx context.Context, collectionName pb.CollectionName) (pb.PartitionList, error)

	DropPartition(ctx context.Context, partitionParam pb.PartitionParam) (pb.Status, error)

	Insert(ctx context.Context, insertParam pb.InsertParam) (pb.VectorIds, error)

	GetVectorsByID(ctx context.Context, identity pb.VectorsIdentity) (pb.VectorsData, error)

	GetVectorIDs(ctx context.Context, param pb.GetVectorIDsParam) (pb.VectorIds, error)

	Search(ctx context.Context, searchParam pb.SearchParam) (*pb.TopKQueryResult, error)

	SearchInFiles(ctx context.Context, searchInFilesParam pb.SearchInFilesParam) (*pb.TopKQueryResult, error)

	Cmd(ctx context.Context, command pb.Command) (pb.StringReply, error)

	DeleteByID(ctx context.Context, param pb.DeleteByIDParam) (pb.Status, error)

	PreloadCollection(ctx context.Context, preloadCollectionParam pb.PreloadCollectionParam) (pb.Status, error)

	ReleaseCollection(ctx context.Context, preloadCollectionParam pb.PreloadCollectionParam) (pb.Status, error)

	Flush(ctx context.Context, param pb.FlushParam) (pb.Status, error)

	Compact(ctx context.Context, name pb.CollectionName) (pb.Status, error)
}

type milvusGrpcClient struct {
	serviceInstance pb.MilvusServiceClient
}

// NewMilvusGrpcClient is the constructor of MilvusGrpcClient
func NewMilvusGrpcClient(client pb.MilvusServiceClient) MilvusGrpcClient {
	return &milvusGrpcClient{client}
}

func (grpcClient *milvusGrpcClient) CreateCollection(ctx context.Context, collectionSchema pb.CollectionSchema) (pb.Status, error) {
	reply, err := grpcClient.serviceInstance.CreateCollection(ctx, &collectionSchema)
	if err != nil {
		return pb.Status{ErrorCode: 0, Reason: ""}, err
	}
	return *reply, err
}

func (grpcClient *milvusGrpcClient) HasCollection(ctx context.Context, collectionName pb.CollectionName) (pb.BoolReply, error) {
	boolReply, err := grpcClient.serviceInstance.HasCollection(ctx, &collectionName)
	if err != nil {
		return pb.BoolReply{Status: nil, BoolReply: false}, err
	}
	return *boolReply, err
}

func (grpcClient *milvusGrpcClient) DescribeCollection(ctx context.Context, collectionName pb.CollectionName) (pb.CollectionSchema, error) {
	collectionSchema, err := grpcClient.serviceInstance.DescribeCollection(ctx, &collectionName)
	if err != nil {
		return pb.CollectionSchema{Status: nil, CollectionName: "", Dimension: 0, IndexFileSize: 0, MetricType: 0}, err
	}
	return *collectionSchema, err
}

func (grpcClient *milvusGrpcClient) CountCollection(ctx context.Context, collectionName pb.CollectionName) (pb.CollectionRowCount, error) {
	count, err := grpcClient.serviceInstance.CountCollection(ctx, &collectionName)
	if err != nil {
		return pb.CollectionRowCount{Status: nil, CollectionRowCount: 0}, err
	}
	return *count, err
}

func (grpcClient *milvusGrpcClient) ShowCollections(ctx context.Context) (pb.CollectionNameList, error) {
	cmd := pb.Command{Cmd: ""}
	collectionNameList, err := grpcClient.serviceInstance.ShowCollections(ctx, &cmd)
	if err != nil {
		return pb.CollectionNameList{Status: nil, CollectionNames: nil}, err
	}
	return *collectionNameList, err
}

func (grpcClient *milvusGrpcClient) ShowCollectionInfo(ctx context.Context, collectionName pb.CollectionName) (pb.CollectionInfo, error) {
	collectionInfo, err := grpcClient.serviceInstance.ShowCollectionInfo(ctx, &collectionName)
	if err != nil {
		return pb.CollectionInfo{Status: nil, JsonInfo: ""}, err
	}
	return *collectionInfo, err
}

func (grpcClient *milvusGrpcClient) DropCollection(ctx context.Context, collectionName pb.CollectionName) (pb.Status, error) {
	status, err := grpcClient.serviceInstance.DropCollection(ctx, &collectionName)
	if err != nil {
		return pb.Status{ErrorCode: 0, Reason: ""}, err
	}
	return *status, err
}

func (grpcClient *milvusGrpcClient) CreateIndex(ctx context.Context, indexParam pb.IndexParam) (pb.Status, error) {
	status, err := grpcClient.serviceInstance.CreateIndex(ctx, &indexParam)
	if err != nil {
		return pb.Status{ErrorCode: 0, Reason: ""}, err
	}
	return *status, err
}

func (grpcClient *milvusGrpcClient) DescribeIndex(ctx context.Context, collectionName pb.CollectionName) (pb.IndexParam, error) {
	indexParam, err := grpcClient.serviceInstance.DescribeIndex(ctx, &collectionName)
	if err != nil {
		return pb.IndexParam{Status: nil, CollectionName: "", IndexType: 0, ExtraParams: nil}, err
	}
	return *indexParam, err
}

func (grpcClient *milvusGrpcClient) DropIndex(ctx context.Context, collectionName pb.CollectionName) (pb.Status, error) {
	status, err := grpcClient.serviceInstance.DropIndex(ctx, &collectionName)
	if err != nil {
		return pb.Status{ErrorCode: 0, Reason: ""}, err
	}
	return *status, err
}

func (grpcClient *milvusGrpcClient) CreatePartition(ctx context.Context, partitionParam pb.PartitionParam) (pb.Status, error) {
	status, err := grpcClient.serviceInstance.CreatePartition(ctx, &partitionParam)
	if err != nil {
		return pb.Status{ErrorCode: 0, Reason: ""}, err
	}
	return *status, err
}

func (grpcClient *milvusGrpcClient) ShowPartitions(ctx context.Context, collectionName pb.CollectionName) (pb.PartitionList, error) {
	status, err := grpcClient.serviceInstance.ShowPartitions(ctx, &collectionName)
	if err != nil {
		return pb.PartitionList{Status: nil, PartitionTagArray: nil}, err
	}
	return *status, err
}

func (grpcClient *milvusGrpcClient) DropPartition(ctx context.Context, partitionParam pb.PartitionParam) (pb.Status, error) {
	status, err := grpcClient.serviceInstance.DropPartition(ctx, &partitionParam)
	if err != nil {
		return pb.Status{ErrorCode: 0, Reason: ""}, err
	}
	return *status, err
}

func (grpcClient *milvusGrpcClient) Insert(ctx context.Context, insertParam pb.InsertParam) (pb.VectorIds, error) {
	vectorIds, err := grpcClient.serviceInstance.Insert(ctx, &insertParam)
	if err != nil {
		return pb.VectorIds{Status: nil, VectorIdArray: nil}, err
	}
	return *vectorIds, err
}

func (grpcClient *milvusGrpcClient) GetVectorsByID(ctx context.Context, identity pb.VectorsIdentity) (pb.VectorsData, error) {
	vectorsData, err := grpcClient.serviceInstance.GetVectorsByID(ctx, &identity)
	if err != nil {
		return pb.VectorsData{Status: nil, VectorsData: nil}, err
	}
	return *vectorsData, err
}

func (grpcClient *milvusGrpcClient) GetVectorIDs(ctx context.Context, param pb.GetVectorIDsParam) (pb.VectorIds, error) {
	status, err := grpcClient.serviceInstance.GetVectorIDs(ctx, &param)
	if err != nil {
		return pb.VectorIds{Status: nil, VectorIdArray: nil}, err
	}
	return *status, err
}

func (grpcClient *milvusGrpcClient) Search(ctx context.Context, searchParam pb.SearchParam) (*pb.TopKQueryResult, error) {
	topkQueryResult, err := grpcClient.serviceInstance.Search(ctx, &searchParam)
	if err != nil {
		return &pb.TopKQueryResult{Status: nil, RowNum: 0, Ids: nil}, err
	}
	return topkQueryResult, err
}

func (grpcClient *milvusGrpcClient) SearchInFiles(ctx context.Context, searchInFilesParam pb.SearchInFilesParam) (*pb.TopKQueryResult, error) {
	topkQueryResult, err := grpcClient.serviceInstance.SearchInFiles(ctx, &searchInFilesParam)
	if err != nil {
		return &pb.TopKQueryResult{Status: nil, RowNum: 0, Ids: nil}, err
	}
	return topkQueryResult, err
}

func (grpcClient *milvusGrpcClient) Cmd(ctx context.Context, command pb.Command) (pb.StringReply, error) {
	stringReply, err := grpcClient.serviceInstance.Cmd(ctx, &command)
	if err != nil {
		return pb.StringReply{Status: nil, StringReply: ""}, err
	}
	return *stringReply, err
}

func (grpcClient *milvusGrpcClient) DeleteByID(ctx context.Context, param pb.DeleteByIDParam) (pb.Status, error) {
	status, err := grpcClient.serviceInstance.DeleteByID(ctx, &param)
	if err != nil {
		return pb.Status{ErrorCode: 0, Reason: ""}, err
	}
	return *status, err
}

func (grpcClient *milvusGrpcClient) PreloadCollection(ctx context.Context, preloadCollectionParam pb.PreloadCollectionParam) (pb.Status, error) {
	status, err := grpcClient.serviceInstance.PreloadCollection(ctx, &preloadCollectionParam)
	if err != nil {
		return pb.Status{ErrorCode: 0, Reason: ""}, err
	}
	return *status, err
}

func (grpcClient *milvusGrpcClient) ReleaseCollection(ctx context.Context, preloadCollectionParam pb.PreloadCollectionParam) (pb.Status, error) {
	status, err := grpcClient.serviceInstance.ReleaseCollection(ctx, &preloadCollectionParam)
	if err != nil {
		return pb.Status{ErrorCode: 0, Reason: ""}, err
	}
	return *status, err
}

func (grpcClient *milvusGrpcClient) Flush(ctx context.Context, param pb.FlushParam) (pb.Status, error) {
	status, err := grpcClient.serviceInstance.Flush(ctx, &param)
	if err != nil {
		return pb.Status{ErrorCode: 0, Reason: ""}, err
	}
	return *status, err
}

func (grpcClient *milvusGrpcClient) Compact(ctx context.Context, collectionName pb.CollectionName) (pb.Status, error) {
	status, err := grpcClient.serviceInstance.Compact(ctx, &collectionName)
	if err != nil {
		return pb.Status{ErrorCode: 0, Reason: ""}, err
	}
	return *status, err
}
