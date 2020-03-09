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
	CreateTable(tableSchema pb.TableSchema) (pb.Status, error)

	HasTable(tableName pb.TableName) (pb.BoolReply, error)

	DescribeTable(tableName pb.TableName) (pb.TableSchema, error)

	CountTable(tableName pb.TableName) (pb.TableRowCount, error)

	ShowTables() (pb.TableNameList, error)

	ShowTableInfos(tableName pb.TableName) (pb.TableInfo, error)

	DropTable(tableName pb.TableName) (pb.Status, error)

	CreateIndex(indexParam pb.IndexParam) (pb.Status, error)

	DescribeIndex(tableName pb.TableName) (pb.IndexParam, error)

	DropIndex(tableName pb.TableName) (pb.Status, error)

	CreatePartition(partitionParam pb.PartitionParam) (pb.Status, error)

	ShowPartitions(tableName pb.TableName) (pb.PartitionList, error)

	DropPartition(partitionParam pb.PartitionParam) (pb.Status, error)

	Insert(insertParam pb.InsertParam) (pb.VectorIds, error)

	GetVectorByID(identity pb.VectorIdentity) (pb.VectorData, error)

	GetVectorIDs(param pb.GetVectorIDsParam) (pb.VectorIds, error)

	Search(searchParam pb.SearchParam) (*pb.TopKQueryResult, error)

	SearchInFiles(searchInFilesParam pb.SearchInFilesParam) (*pb.TopKQueryResult, error)

	Cmd(command pb.Command) (pb.StringReply, error)

	DeleteByID(param pb.DeleteByIDParam) (pb.Status, error)

	PreloadTable(tableName pb.TableName) (pb.Status, error)

	Flush(param pb.FlushParam) (pb.Status, error)

	Compact(name pb.TableName) (pb.Status, error)
}

type milvusGrpcClient struct {
	serviceInstance pb.MilvusServiceClient
}

// NewMilvusGrpcClient is the constructor of MilvusGrpcClient
func NewMilvusGrpcClient(client pb.MilvusServiceClient) MilvusGrpcClient {
	return &milvusGrpcClient{client}
}

func (grpcClient *milvusGrpcClient) CreateTable(tableSchema pb.TableSchema) (pb.Status, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	reply, err := grpcClient.serviceInstance.CreateTable(ctx, &tableSchema)
	return *reply, err
}

func (grpcClient *milvusGrpcClient) HasTable(tableName pb.TableName) (pb.BoolReply, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	boolReply, err := grpcClient.serviceInstance.HasTable(ctx, &tableName)
	return *boolReply, err
}

func (grpcClient *milvusGrpcClient) DescribeTable(tableName pb.TableName) (pb.TableSchema, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	tableSchema, err := grpcClient.serviceInstance.DescribeTable(ctx, &tableName)
	return *tableSchema, err
}

func (grpcClient *milvusGrpcClient) CountTable(tableName pb.TableName) (pb.TableRowCount, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	count, err := grpcClient.serviceInstance.CountTable(ctx, &tableName)
	return *count, err
}

func (grpcClient *milvusGrpcClient) ShowTables() (pb.TableNameList, error) {
	cmd := pb.Command{"", struct{}{}, nil, 0}
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	tableNameList, err := grpcClient.serviceInstance.ShowTables(ctx, &cmd)
	return *tableNameList, err
}

func (grpcClient *milvusGrpcClient) ShowTableInfos(tableName pb.TableName) (pb.TableInfo, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	tableInfo, err := grpcClient.serviceInstance.ShowTableInfo(ctx, &tableName)
	return *tableInfo, err
}

func (grpcClient *milvusGrpcClient) DropTable(tableName pb.TableName) (pb.Status, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	status, err := grpcClient.serviceInstance.DropTable(ctx, &tableName)
	return *status, err
}

func (grpcClient *milvusGrpcClient) CreateIndex(indexParam pb.IndexParam) (pb.Status, error) {
	ctx := context.Background()
	status, err := grpcClient.serviceInstance.CreateIndex(ctx, &indexParam)
	return *status, err
}

func (grpcClient *milvusGrpcClient) DescribeIndex(tableName pb.TableName) (pb.IndexParam, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	indexParam, err := grpcClient.serviceInstance.DescribeIndex(ctx, &tableName)
	return *indexParam, err
}

func (grpcClient *milvusGrpcClient) DropIndex(tableName pb.TableName) (pb.Status, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	status, err := grpcClient.serviceInstance.DropIndex(ctx, &tableName)
	return *status, err
}

func (grpcClient *milvusGrpcClient) CreatePartition(partitionParam pb.PartitionParam) (pb.Status, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	status, err := grpcClient.serviceInstance.CreatePartition(ctx, &partitionParam)
	return *status, err
}

func (grpcClient *milvusGrpcClient) ShowPartitions(tableName pb.TableName) (pb.PartitionList, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	status, err := grpcClient.serviceInstance.ShowPartitions(ctx, &tableName)
	return *status, err
}

func (grpcClient *milvusGrpcClient) DropPartition(partitionParam pb.PartitionParam) (pb.Status, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	status, err := grpcClient.serviceInstance.DropPartition(ctx, &partitionParam)
	return *status, err
}

func (grpcClient *milvusGrpcClient) Insert(insertParam pb.InsertParam) (pb.VectorIds, error) {
	ctx := context.Background()
	vectorIds, err := grpcClient.serviceInstance.Insert(ctx, &insertParam)
	return *vectorIds, err
}

func (grpcClient *milvusGrpcClient) GetVectorByID(identity pb.VectorIdentity) (pb.VectorData, error) {
	ctx := context.Background()
	status, err := grpcClient.serviceInstance.GetVectorByID(ctx, &identity)
	return *status, err
}

func (grpcClient *milvusGrpcClient) GetVectorIDs(param pb.GetVectorIDsParam) (pb.VectorIds, error) {
	ctx := context.Background()
	status, err := grpcClient.serviceInstance.GetVectorIDs(ctx, &param)
	return *status, err
}

func (grpcClient *milvusGrpcClient) Search(searchParam pb.SearchParam) (*pb.TopKQueryResult, error) {
	ctx := context.Background()
	topkQueryResult, err := grpcClient.serviceInstance.Search(ctx, &searchParam)
	return topkQueryResult, err
}

func (grpcClient *milvusGrpcClient) SearchInFiles(searchInFilesParam pb.SearchInFilesParam) (*pb.TopKQueryResult, error) {
	ctx := context.Background()
	topkQueryResult, err := grpcClient.serviceInstance.SearchInFiles(ctx, &searchInFilesParam)
	return topkQueryResult, err
}

func (grpcClient *milvusGrpcClient) Cmd(command pb.Command) (pb.StringReply, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	stringReply, err := grpcClient.serviceInstance.Cmd(ctx, &command)
	return *stringReply, err
}

func (grpcClient *milvusGrpcClient) DeleteByID(param pb.DeleteByIDParam) (pb.Status, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	status, err := grpcClient.serviceInstance.DeleteByID(ctx, &param)
	return *status, err
}

func (grpcClient *milvusGrpcClient) PreloadTable(tableName pb.TableName) (pb.Status, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	status, err := grpcClient.serviceInstance.PreloadTable(ctx, &tableName)
	return *status, err
}

func (grpcClient *milvusGrpcClient) Flush(param pb.FlushParam) (pb.Status, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	status, err := grpcClient.serviceInstance.Flush(ctx, &param)
	return *status, err
}

func (grpcClient *milvusGrpcClient) Compact(tableName pb.TableName) (pb.Status, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	status, err := grpcClient.serviceInstance.Compact(ctx, &tableName)
	return *status, err
}
