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

package milvus

import (
	"context"
	pb "github.com/milvus-io/milvus-sdk-go/milvus/grpc/gen"
	"google.golang.org/grpc"
	"log"
	"time"
)

type Milvusclient struct {
	Instance MilvusGrpcClient
}

// NewMilvusClient is the constructor of MilvusClient
func NewMilvusClient(client MilvusGrpcClient) MilvusClient {
	return &Milvusclient{client}
}

func (client *Milvusclient) GetClientVersion() string {
	return clientVersion
}

func (client *Milvusclient) Connect(connectParam ConnectParam) Status {
	var opts []grpc.DialOption
	opts = append(opts, grpc.WithInsecure())
	opts = append(opts, grpc.WithBlock())

	serverAddr := connectParam.IPAddress + ":" + connectParam.Port

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	conn, err := grpc.DialContext(ctx, serverAddr, opts...)
	if err != nil {
		log.Fatalf("fail to dial: %v", err)
	}

	milvusclient := pb.NewMilvusServiceClient(conn)

	milvusGrpcClient := NewMilvusGrpcClient(milvusclient)

	client.Instance = milvusGrpcClient

	return status{0, ""}
}

func (client *Milvusclient) IsConnected() bool {
	return client.Instance != nil
}

func (client *Milvusclient) Disconnect() Status {
	client.Instance = nil
	return status{0, ""}
}

func (client *Milvusclient) CreateTable(tableSchema TableSchema) Status {
	grpcTableSchema := pb.TableSchema{nil, tableSchema.TableName, tableSchema.Dimension,
		tableSchema.IndexFileSize, int32(tableSchema.MetricType), struct{}{}, nil, 0}
	grpcStatus, err := client.Instance.CreateTable(grpcTableSchema)
	if err != nil {
		return status{int64(RPCFailed), err.Error()}
	}
	errorCode := int64(grpcStatus.ErrorCode)
	return status{errorCode, grpcStatus.Reason}
}

func (client *Milvusclient) HasTable(tableName string) (Status, bool) {
	grpcTableName := pb.TableName{tableName, struct{}{}, nil, 0}
	boolReply, err := client.Instance.HasTable(grpcTableName)
	if err != nil {
		return status{int64(RPCFailed), err.Error()}, boolReply.GetBoolReply()
	}
	return status{int64(boolReply.GetStatus().GetErrorCode()), boolReply.GetStatus().GetReason()}, boolReply.GetBoolReply()
}

func (client *Milvusclient) DropTable(tableName string) Status {
	grpcTableName := pb.TableName{tableName, struct{}{}, nil, 0}
	grpcStatus, err := client.Instance.DropTable(grpcTableName)
	if err != nil {
		return status{int64(RPCFailed), err.Error()}
	}
	errorCode := int64(grpcStatus.ErrorCode)
	return status{errorCode, grpcStatus.Reason}
}

func (client *Milvusclient) CreateIndex(indexParam *IndexParam) Status {
	index := pb.Index{int32(indexParam.IndexType), int32(indexParam.Nlist),
		struct{}{}, nil, 0}
	grpcIndexParam := pb.IndexParam{nil, indexParam.TableName, &index,
		struct{}{}, nil, 0}
	grpcStatus, err := client.Instance.CreateIndex(grpcIndexParam)
	if err != nil {
		return status{int64(RPCFailed), err.Error()}
	}
	return status{int64(grpcStatus.ErrorCode), grpcStatus.Reason}
}

////////////////////////////////////////////////////////////////////////////

func (client *Milvusclient) Insert(insertParam *InsertParam) Status {
	var i int
	var rowRecordArray = make([]*pb.RowRecord, len(insertParam.RecordArray))
	for i = 0; i < len(insertParam.RecordArray); i++ {
		rowRecord := pb.RowRecord{insertParam.RecordArray[i], struct{}{}, nil, 0}
		rowRecordArray[i] = &rowRecord
	}
	grpcInsertParam := pb.InsertParam{insertParam.TableName, rowRecordArray, insertParam.IDArray,
		insertParam.PartitionTag, struct{}{}, nil, 0}
	vectorIds, err := client.Instance.Insert(grpcInsertParam)
	if err != nil {
		return status{int64(RPCFailed), err.Error()}
	}
	insertParam.IDArray = vectorIds.VectorIdArray
	return status{int64(vectorIds.Status.ErrorCode), vectorIds.Status.Reason}
}

////////////////////////////////////////////////////////////////////////////

func (client *Milvusclient) Search(searchParam SearchParam) (Status, TopkQueryResult) {
	var queryRecordArray = make([]*pb.RowRecord, len(searchParam.QueryVectors))
	var i, j int64
	for i = 0; i < int64(len(searchParam.QueryVectors)); i++ {
		rowRecord := pb.RowRecord{searchParam.QueryVectors[i], struct{}{}, nil, 0}
		queryRecordArray[i] = &rowRecord
	}
	grpcSearchParam := pb.SearchParam{searchParam.TableName, queryRecordArray, nil,
		searchParam.Topk, searchParam.Nprobe, searchParam.PartitionTag, struct{}{}, nil, 0}
	topkQueryResult, err := client.Instance.Search(grpcSearchParam)
	if err != nil {
		return status{int64(RPCFailed), err.Error()}, TopkQueryResult{nil}
	}
	nq := topkQueryResult.GetRowNum()
	var result = make([]QueryResult, nq)
	for i = 0; i < nq; i++ {
		topk := int64(len(topkQueryResult.GetIds())) / nq
		result[i].Ids = make([]int64, topk)
		result[i].Distances = make([]float32, topk)
		for j = 0; j < topk; j++ {
			result[i].Ids[j] = topkQueryResult.GetIds()[i*nq+j]
			result[i].Distances[j] = topkQueryResult.GetDistances()[i*nq+j]
		}
	}
	return status{int64(topkQueryResult.Status.ErrorCode), topkQueryResult.Status.Reason},
		TopkQueryResult{result}
}

func (client *Milvusclient) DescribeTable(tableName string) (Status, TableSchema) {
	grpcTableName := pb.TableName{tableName, struct{}{}, nil, 0}
	tableSchema, err := client.Instance.DescribeTable(grpcTableName)
	if err != nil {
		return status{int64(RPCFailed), err.Error()}, TableSchema{"", 0, 0, 0}
	}
	return status{int64(tableSchema.GetStatus().GetErrorCode()), tableSchema.Status.Reason},
		TableSchema{tableSchema.GetTableName(), tableSchema.GetDimension(), tableSchema.GetIndexFileSize(), int64(tableSchema.GetMetricType())}
}

func (client *Milvusclient) CountTable(tableName string) (Status, int64) {
	grpcTableName := pb.TableName{tableName, struct{}{}, nil, 0}
	rowCount, err := client.Instance.CountTable(grpcTableName)
	if err != nil {
		return status{int64(RPCFailed), err.Error()}, 0
	}
	return status{int64(rowCount.GetStatus().GetErrorCode()), rowCount.GetStatus().GetReason()}, rowCount.GetTableRowCount()

}

func (client *Milvusclient) ShowTables() (Status, []string) {
	tableNameList, err := client.Instance.ShowTable()
	if err != nil {
		return status{int64(RPCFailed), err.Error()}, nil
	}
	return status{int64(tableNameList.GetStatus().GetErrorCode()), tableNameList.GetStatus().GetReason()}, tableNameList.GetTableNames()
}

func (client *Milvusclient) ServerVersion() (Status, string) {
	command := pb.Command{"version", struct{}{}, nil, 0}
	serverVersion, err := client.Instance.Cmd(command)
	if err != nil {
		return status{int64(RPCFailed), err.Error()}, ""
	}
	return status{int64(serverVersion.GetStatus().GetErrorCode()), serverVersion.GetStatus().GetReason()}, serverVersion.GetStringReply()
}

func (client *Milvusclient) ServerStatus() (Status, string) {
	if client.Instance == nil {
		return status{int64(0), ""}, "not connect to server"
	}
	command := pb.Command{"", struct{}{}, nil, 0}
	serverStatus, err := client.Instance.Cmd(command)
	if err != nil {
		return status{int64(RPCFailed), err.Error()}, ""
	}
	return status{int64(serverStatus.GetStatus().GetErrorCode()), serverStatus.GetStatus().GetReason()}, serverStatus.GetStringReply()
}

func (client *Milvusclient) PreloadTable(tableName string) Status {
	grpcTableName := pb.TableName{tableName, struct{}{}, nil, 0}
	grpcStatus, err := client.Instance.PreloadTable(grpcTableName)
	if err != nil {
		return status{int64(RPCFailed), err.Error()}
	}
	return status{int64(grpcStatus.GetErrorCode()), grpcStatus.GetReason()}
}

func (client *Milvusclient) DescribeIndex(tableName string) (Status, IndexParam) {
	grpcTableName := pb.TableName{tableName, struct{}{}, nil, 0}
	indexParam, err := client.Instance.DescribeIndex(grpcTableName)
	if err != nil {
		return status{int64(RPCFailed), err.Error()}, IndexParam{"", 0, 0}
	}
	return status{int64(indexParam.GetStatus().GetErrorCode()), indexParam.GetStatus().GetReason()},
		IndexParam{indexParam.GetTableName(), IndexType(indexParam.GetIndex().GetIndexType()), int64(indexParam.GetIndex().GetNlist())}
}

func (client *Milvusclient) DropIndex(tableName string) Status {
	grpcTableName := pb.TableName{tableName, struct{}{}, nil, 0}
	grpcStatus, err := client.Instance.DropIndex(grpcTableName)
	if err != nil {
		return status{int64(RPCFailed), err.Error()}
	}
	return status{int64(grpcStatus.GetErrorCode()), grpcStatus.GetReason()}
}

func (client *Milvusclient) CreatePartition(partitionParam PartitionParam) Status {
	grpcPartitionParam := pb.PartitionParam{partitionParam.TableName, partitionParam.PartitionName,
		partitionParam.PartitionTag, struct{}{}, nil, 0}
	grpcStatus, err := client.Instance.CreatePartition(grpcPartitionParam)
	if err != nil {
		return status{int64(RPCFailed), err.Error()}
	}
	return status{int64(grpcStatus.GetErrorCode()), grpcStatus.GetReason()}
}

func (client *Milvusclient) ShowPartitions(tableName string) (Status, []PartitionParam) {
	grpcTableName := pb.TableName{tableName, struct{}{}, nil, 0}
	grpcPartitionList, err := client.Instance.ShowPartitions(grpcTableName)
	if err != nil {
		return status{int64(RPCFailed), err.Error()}, nil
	}
	var partitionList = make([]PartitionParam, len(grpcPartitionList.GetPartitionArray()))
	var i int
	for i = 0; i < len(grpcPartitionList.GetPartitionArray()); i++ {
		partitionList[i].TableName = grpcPartitionList.GetPartitionArray()[i].TableName
		partitionList[i].PartitionTag = grpcPartitionList.GetPartitionArray()[i].Tag
		partitionList[i].PartitionName = grpcPartitionList.GetPartitionArray()[i].PartitionName
	}
	return status{int64(grpcPartitionList.GetStatus().GetErrorCode()), grpcPartitionList.GetStatus().GetReason()}, partitionList
}

func (client *Milvusclient) DropPartition(partitionParam PartitionParam) Status {
	grpcPartitionParam := pb.PartitionParam{partitionParam.TableName, partitionParam.PartitionName,
		partitionParam.PartitionTag, struct{}{}, nil, 0}
	grpcStatus, err := client.Instance.DropPartition(grpcPartitionParam)
	if err != nil {
		return status{int64(RPCFailed), err.Error()}
	}
	return status{int64(grpcStatus.GetErrorCode()), grpcStatus.Reason}
}

func (client *Milvusclient) GetConfig(nodeName string) (Status, string) {
	command := pb.Command{"get_config " + nodeName, struct{}{}, nil, 0}
	configInfo, err := client.Instance.Cmd(command)
	if err != nil {
		return status{int64(RPCFailed), err.Error()}, ""
	}
	return status{int64(configInfo.GetStatus().GetErrorCode()), configInfo.GetStatus().GetReason()}, configInfo.GetStringReply()
}

func (client *Milvusclient) SetConfig(nodeName string, value string) Status {
	command := pb.Command{"set_config " + nodeName + " " + value, struct{}{}, nil, 0}
	reply, err := client.Instance.Cmd(command)
	if err != nil {
		return status{int64(RPCFailed), err.Error()}
	}
	return status{int64(reply.GetStatus().GetErrorCode()), reply.GetStatus().GetReason()}
}
