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
	"google.golang.org/grpc"
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

func (client *Milvusclient) Connect(connectParam ConnectParam) error {
	var opts []grpc.DialOption
	opts = append(opts, grpc.WithInsecure())
	opts = append(opts, grpc.WithBlock())

	serverAddr := connectParam.IPAddress + ":" + connectParam.Port

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	conn, err := grpc.DialContext(ctx, serverAddr, opts...)
	if err != nil {
		return err
	}

	milvusclient := pb.NewMilvusServiceClient(conn)

	milvusGrpcClient := NewMilvusGrpcClient(milvusclient)

	client.Instance = milvusGrpcClient

	return nil
}

func (client *Milvusclient) IsConnected() bool {
	return client.Instance != nil
}

func (client *Milvusclient) Disconnect() error {
	client.Instance = nil
	return nil
}

func (client *Milvusclient) CreateTable(tableSchema TableSchema) (Status, error) {
	keyValuePair := make([]*pb.KeyValuePair, len(tableSchema.ExtraParams))
	var i int
	for i = 0; i < len(tableSchema.ExtraParams); i++ {
		pair := pb.KeyValuePair{tableSchema.ExtraParams[i].Key, tableSchema.ExtraParams[i].Value, struct{}{}, nil, 0}
		keyValuePair[i] = &pair
	}
	grpcTableSchema := pb.TableSchema{nil, tableSchema.TableName, tableSchema.Dimension,
		tableSchema.IndexFileSize, int32(tableSchema.MetricType), keyValuePair, struct{}{}, nil, 0}
	grpcStatus, err := client.Instance.CreateTable(grpcTableSchema)
	if err != nil {
		return nil, err
	}
	errorCode := int64(grpcStatus.ErrorCode)
	return status{errorCode, grpcStatus.Reason}, err
}

func (client *Milvusclient) HasTable(tableName string) (bool, Status, error) {
	grpcTableName := pb.TableName{tableName, struct{}{}, nil, 0}
	boolReply, err := client.Instance.HasTable(grpcTableName)
	if err != nil {
		return false, nil, err
	}
	return boolReply.GetBoolReply(), status{int64(boolReply.GetStatus().GetErrorCode()), boolReply.GetStatus().GetReason()}, err
}

func (client *Milvusclient) DropTable(tableName string) (Status, error) {
	grpcTableName := pb.TableName{tableName, struct{}{}, nil, 0}
	grpcStatus, err := client.Instance.DropTable(grpcTableName)
	if err != nil {
		return nil, err
	}
	errorCode := int64(grpcStatus.ErrorCode)
	return status{errorCode, grpcStatus.Reason}, err
}

func (client *Milvusclient) CreateIndex(indexParam *IndexParam) (Status, error) {
	keyValuePair := make([]*pb.KeyValuePair, len(indexParam.ExtraParams))
	var i int
	for i = 0; i < len(indexParam.ExtraParams); i++ {
		pair := pb.KeyValuePair{indexParam.ExtraParams[i].Key, indexParam.ExtraParams[i].Value, struct{}{}, nil, 0}
		keyValuePair[i] = &pair
	}
	grpcIndexParam := pb.IndexParam{nil, indexParam.TableName, int32(indexParam.IndexType), keyValuePair,
		struct{}{}, nil, 0}
	grpcStatus, err := client.Instance.CreateIndex(grpcIndexParam)
	if err != nil {
		return nil, err
	}
	return status{int64(grpcStatus.ErrorCode), grpcStatus.Reason}, err
}

////////////////////////////////////////////////////////////////////////////

func (client *Milvusclient) Insert(insertParam *InsertParam) (Status, error) {
	var i int
	var rowRecordArray = make([]*pb.RowRecord, len(insertParam.RecordArray))
	for i = 0; i < len(insertParam.RecordArray); i++ {
		rowRecord := pb.RowRecord{insertParam.RecordArray[i].FloatData, insertParam.RecordArray[i].BinaryData, struct{}{}, nil, 0}
		rowRecordArray[i] = &rowRecord
	}

	keyValuePair := make([]*pb.KeyValuePair, len(insertParam.ExtraParams))
	for i = 0; i < len(insertParam.ExtraParams); i++ {
		pair := pb.KeyValuePair{insertParam.ExtraParams[i].Key, insertParam.ExtraParams[i].Value, struct{}{}, nil, 0}
		keyValuePair[i] = &pair
	}

	grpcInsertParam := pb.InsertParam{insertParam.TableName, rowRecordArray, insertParam.IDArray,
		insertParam.PartitionTag, keyValuePair, struct{}{}, nil, 0}
	vectorIds, err := client.Instance.Insert(grpcInsertParam)
	if err != nil {
		return nil, err
	}
	insertParam.IDArray = vectorIds.VectorIdArray
	return status{int64(vectorIds.Status.ErrorCode), vectorIds.Status.Reason}, err
}

////////////////////////////////////////////////////////////////////////////

func (client *Milvusclient) GetVectorByID(tableName string, vector_id int64) (RowRecord, Status, error) {
	grpcIdentity := pb.VectorIdentity{tableName, vector_id, struct{}{}, nil, 0}
	grpcVectorData, err := client.Instance.GetVectorByID(grpcIdentity)
	if err != nil {
		return RowRecord{nil, nil}, nil, err
	}
	if grpcVectorData.VectorData != nil {
		return RowRecord{grpcVectorData.VectorData.FloatData, grpcVectorData.VectorData.BinaryData},
			status{int64(grpcVectorData.Status.ErrorCode), grpcVectorData.Status.Reason}, err
	}
	return RowRecord{nil, nil}, status{int64(grpcVectorData.Status.ErrorCode), grpcVectorData.Status.Reason}, err
}

////////////////////////////////////////////////////////////////////////////

func (client *Milvusclient) GetVectorIDs(getVectorIDsParam GetVectorIDsParam) ([]int64, Status, error) {
	grpcParam := pb.GetVectorIDsParam{getVectorIDsParam.TableName, getVectorIDsParam.SegmentName, struct{}{}, nil, 0}
	vectorIDs, err := client.Instance.GetVectorIDs(grpcParam)
	if err != nil {
		return nil, nil, err
	}
	return vectorIDs.VectorIdArray, status{int64(vectorIDs.Status.ErrorCode), vectorIDs.Status.Reason}, err
}

////////////////////////////////////////////////////////////////////////////

func (client *Milvusclient) Search(searchParam SearchParam) (TopkQueryResult, Status, error) {
	var queryRecordArray = make([]*pb.RowRecord, len(searchParam.QueryVectors))
	var i, j int64
	for i = 0; i < int64(len(searchParam.QueryVectors)); i++ {
		rowRecord := pb.RowRecord{searchParam.QueryVectors[i].FloatData, searchParam.QueryVectors[i].BinaryData, struct{}{}, nil, 0}
		queryRecordArray[i] = &rowRecord
	}

	keyValuePair := make([]*pb.KeyValuePair, len(searchParam.ExtraParams))
	for i = 0; i < int64(len(searchParam.ExtraParams)); i++ {
		pair := pb.KeyValuePair{searchParam.ExtraParams[i].Key, searchParam.ExtraParams[i].Value, struct{}{}, nil, 0}
		keyValuePair[i] = &pair
	}

	grpcSearchParam := pb.SearchParam{searchParam.TableName, searchParam.PartitionTag, queryRecordArray,
		searchParam.Topk, keyValuePair, struct{}{}, nil, 0}
	topkQueryResult, err := client.Instance.Search(grpcSearchParam)
	if err != nil {
		return TopkQueryResult{nil}, nil, err
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
	return TopkQueryResult{result}, status{int64(topkQueryResult.Status.ErrorCode), topkQueryResult.Status.Reason}, nil
}

////////////////////////////////////////////////////////////////////////////

func (client *Milvusclient) SearchByID(searchByIDParam SearchByIDParam) (TopkQueryResult, Status, error) {
	var i, j int64
	keyValuePair := make([]*pb.KeyValuePair, len(searchByIDParam.ExtraParams))
	for i = 0; i < int64(len(searchByIDParam.ExtraParams)); i++ {
		pair := pb.KeyValuePair{searchByIDParam.ExtraParams[i].Key, searchByIDParam.ExtraParams[i].Value, struct{}{}, nil, 0}
		keyValuePair[i] = &pair
	}

	grpcParam := pb.SearchByIDParam{searchByIDParam.TableName, searchByIDParam.PartitionTag, searchByIDParam.Id, searchByIDParam.Topk,
		keyValuePair, struct{}{}, nil, 0}
	topkQueryResult, err := client.Instance.SearchByID(grpcParam)
	if err != nil {
		return TopkQueryResult{nil}, nil, err
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
	return TopkQueryResult{result}, status{int64(topkQueryResult.Status.ErrorCode), topkQueryResult.Status.Reason}, nil
}

////////////////////////////////////////////////////////////////////////////

func (client *Milvusclient) DeleteByID(tableName string, id_array []int64) (Status, error) {
	grpcParam := pb.DeleteByIDParam{tableName, id_array, struct{}{}, nil, 0}
	grpcStatus, err := client.Instance.DeleteByID(grpcParam)
	if err != nil {
		return nil, err
	}
	return status{int64(grpcStatus.GetErrorCode()), grpcStatus.GetReason()}, err
}

////////////////////////////////////////////////////////////////////////////

func (client *Milvusclient) DescribeTable(tableName string) (TableSchema, Status, error) {
	grpcTableName := pb.TableName{tableName, struct{}{}, nil, 0}
	tableSchema, err := client.Instance.DescribeTable(grpcTableName)
	if err != nil {
		return TableSchema{"", 0, 0, 0, nil}, nil, err
	}
	return TableSchema{tableSchema.GetTableName(), tableSchema.GetDimension(), tableSchema.GetIndexFileSize(), int64(tableSchema.GetMetricType()), nil},
		status{int64(tableSchema.GetStatus().GetErrorCode()), tableSchema.Status.Reason}, err
}

////////////////////////////////////////////////////////////////////////////

func (client *Milvusclient) CountTable(tableName string) (int64, Status, error) {
	grpcTableName := pb.TableName{tableName, struct{}{}, nil, 0}
	rowCount, err := client.Instance.CountTable(grpcTableName)
	if err != nil {
		return 0, nil, err
	}
	return rowCount.GetTableRowCount(), status{int64(rowCount.GetStatus().GetErrorCode()),
		rowCount.GetStatus().GetReason()}, err
}

////////////////////////////////////////////////////////////////////////////

func (client *Milvusclient) ShowTables() ([]string, Status, error) {
	tableNameList, err := client.Instance.ShowTables()
	if err != nil {
		return nil, nil, err
	}
	return tableNameList.GetTableNames(), status{int64(tableNameList.GetStatus().GetErrorCode()), tableNameList.GetStatus().GetReason()}, err
}

////////////////////////////////////////////////////////////////////////////

func (client *Milvusclient) ShowTableInfo(tableName string) (TableInfo, Status, error) {
	grpcTableName := pb.TableName{tableName, struct{}{}, nil, 0}
	grpcTableInfo, err := client.Instance.ShowTableInfos(grpcTableName)
	if err != nil {
		return TableInfo{0, nil}, nil, err
	}
	partitionStats := make([]PartitionStat, len(grpcTableInfo.PartitionsStat))
	var i, j int
	for i = 0; i < len(grpcTableInfo.PartitionsStat); i++ {
		partitionStats[i].Tag = grpcTableInfo.PartitionsStat[i].Tag
		partitionStats[i].RowCount = grpcTableInfo.PartitionsStat[i].TotalRowCount
		segmentStat := make([]SegmentStat, len(grpcTableInfo.PartitionsStat[i].SegmentsStat))
		for j = 0; j < len(grpcTableInfo.GetPartitionsStat()[i].GetSegmentsStat()); j++ {
			segmentStat[j] = SegmentStat{grpcTableInfo.PartitionsStat[i].SegmentsStat[j].SegmentName,
				grpcTableInfo.PartitionsStat[i].SegmentsStat[j].RowCount,
				grpcTableInfo.PartitionsStat[i].SegmentsStat[j].IndexName,
				grpcTableInfo.PartitionsStat[i].SegmentsStat[j].DataSize}
		}
		partitionStats[i].SegmentsStat = segmentStat
	}
	tableInfo := TableInfo{grpcTableInfo.TotalRowCount, partitionStats}
	return tableInfo, status{int64(grpcTableInfo.Status.ErrorCode), grpcTableInfo.Status.Reason}, err
}

////////////////////////////////////////////////////////////////////////////

func (client *Milvusclient) ServerVersion() (string, Status, error) {
	command := pb.Command{"version", struct{}{}, nil, 0}
	serverVersion, err := client.Instance.Cmd(command)
	if err != nil {
		return "", nil, err
	}
	return serverVersion.GetStringReply(), status{int64(serverVersion.GetStatus().GetErrorCode()), serverVersion.GetStatus().GetReason()}, err
}

////////////////////////////////////////////////////////////////////////////

func (client *Milvusclient) ServerStatus() (string, Status, error) {
	if client.Instance == nil {
		return "not connect to server", status{int64(0), ""}, nil
	}
	command := pb.Command{"", struct{}{}, nil, 0}
	serverStatus, err := client.Instance.Cmd(command)
	if err != nil {
		return "connection lost", nil, err
	}
	return "server alive", status{int64(serverStatus.GetStatus().GetErrorCode()), serverStatus.GetStatus().GetReason()}, err
}

////////////////////////////////////////////////////////////////////////////

func (client *Milvusclient) PreloadTable(tableName string) (Status, error) {
	grpcTableName := pb.TableName{tableName, struct{}{}, nil, 0}
	grpcStatus, err := client.Instance.PreloadTable(grpcTableName)
	if err != nil {
		return nil, err
	}
	return status{int64(grpcStatus.GetErrorCode()), grpcStatus.GetReason()}, err
}

////////////////////////////////////////////////////////////////////////////

func (client *Milvusclient) DescribeIndex(tableName string) (IndexParam, Status, error) {
	grpcTableName := pb.TableName{tableName, struct{}{}, nil, 0}
	indexParam, err := client.Instance.DescribeIndex(grpcTableName)
	if err != nil {
		return IndexParam{"", 0, nil}, nil, err
	}
	return IndexParam{indexParam.GetTableName(), IndexType(indexParam.IndexType), nil},
		status{int64(indexParam.GetStatus().GetErrorCode()), indexParam.GetStatus().GetReason()}, err
}

////////////////////////////////////////////////////////////////////////////

func (client *Milvusclient) DropIndex(tableName string) (Status, error) {
	grpcTableName := pb.TableName{tableName, struct{}{}, nil, 0}
	grpcStatus, err := client.Instance.DropIndex(grpcTableName)
	if err != nil {
		return nil, err
	}
	return status{int64(grpcStatus.GetErrorCode()), grpcStatus.GetReason()}, err
}

////////////////////////////////////////////////////////////////////////////

func (client *Milvusclient) CreatePartition(partitionParam PartitionParam) (Status, error) {
	grpcPartitionParam := pb.PartitionParam{partitionParam.TableName,
		partitionParam.PartitionTag, struct{}{}, nil, 0}
	grpcStatus, err := client.Instance.CreatePartition(grpcPartitionParam)
	if err != nil {
		return nil, err
	}
	return status{int64(grpcStatus.GetErrorCode()), grpcStatus.GetReason()}, err
}

////////////////////////////////////////////////////////////////////////////

func (client *Milvusclient) ShowPartitions(tableName string) ([]PartitionParam, Status, error) {
	grpcTableName := pb.TableName{tableName, struct{}{}, nil, 0}
	grpcPartitionList, err := client.Instance.ShowPartitions(grpcTableName)
	if err != nil {
		return nil, status{int64(RPCFailed), err.Error()}, err
	}
	var partitionList = make([]PartitionParam, len(grpcPartitionList.GetPartitionTagArray()))
	var i int
	for i = 0; i < len(grpcPartitionList.GetPartitionTagArray()); i++ {
		partitionList[i].TableName = grpcPartitionList.GetPartitionTagArray()[i]
	}
	return partitionList, status{int64(grpcPartitionList.GetStatus().GetErrorCode()), grpcPartitionList.GetStatus().GetReason()}, err
}

////////////////////////////////////////////////////////////////////////////

func (client *Milvusclient) DropPartition(partitionParam PartitionParam) (Status, error) {
	grpcPartitionParam := pb.PartitionParam{partitionParam.TableName,
		partitionParam.PartitionTag, struct{}{}, nil, 0}
	grpcStatus, err := client.Instance.DropPartition(grpcPartitionParam)
	if err != nil {
		return nil, err
	}
	return status{int64(grpcStatus.GetErrorCode()), grpcStatus.Reason}, err
}

////////////////////////////////////////////////////////////////////////////

func (client *Milvusclient) GetConfig(nodeName string) (string, Status, error) {
	command := pb.Command{"get_config " + nodeName, struct{}{}, nil, 0}
	configInfo, err := client.Instance.Cmd(command)
	if err != nil {
		return "", nil, err
	}
	return configInfo.GetStringReply(), status{int64(configInfo.GetStatus().GetErrorCode()), configInfo.GetStatus().GetReason()}, err
}

////////////////////////////////////////////////////////////////////////////

func (client *Milvusclient) SetConfig(nodeName string, value string) (Status, error) {
	command := pb.Command{"set_config " + nodeName + " " + value, struct{}{}, nil, 0}
	reply, err := client.Instance.Cmd(command)
	if err != nil {
		return nil, err
	}
	return status{int64(reply.GetStatus().GetErrorCode()), reply.GetStatus().GetReason()}, err
}

////////////////////////////////////////////////////////////////////////////

func (client *Milvusclient) Flush(tableNameArray []string) (Status, error) {
	grpcParam := pb.FlushParam{tableNameArray, struct{}{}, nil, 0}
	grpcStatus, err := client.Instance.Flush(grpcParam)
	if err != nil {
		return nil, err
	}
	return status{int64(grpcStatus.GetErrorCode()), grpcStatus.GetReason()}, err
}

////////////////////////////////////////////////////////////////////////////

func (client *Milvusclient) Compact(tableName string) (Status, error) {
	grpcParam := pb.TableName{tableName, struct{}{}, nil, 0}
	grpcStatus, err := client.Instance.Compact(grpcParam)
	if err != nil {
		return nil, err
	}
	return status{int64(grpcStatus.GetErrorCode()), grpcStatus.GetReason()}, err
}
