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

func (client *Milvusclient) CreateCollection(collectionParam CollectionParam) (Status, error) {
	grpcCollectionSchema := pb.CollectionSchema{nil, collectionParam.CollectionName, collectionParam.Dimension,
		collectionParam.IndexFileSize, int32(collectionParam.MetricType), nil, struct{}{}, nil, 0}
	grpcStatus, err := client.Instance.CreateCollection(grpcCollectionSchema)
	if err != nil {
		return nil, err
	}
	errorCode := int64(grpcStatus.ErrorCode)
	return status{errorCode, grpcStatus.Reason}, err
}

func (client *Milvusclient) HasCollection(collectionName string) (bool, Status, error) {
	grpcCollectionName := pb.CollectionName{collectionName, struct{}{}, nil, 0}
	boolReply, err := client.Instance.HasCollection(grpcCollectionName)
	if err != nil {
		return false, nil, err
	}
	return boolReply.GetBoolReply(), status{int64(boolReply.GetStatus().GetErrorCode()), boolReply.GetStatus().GetReason()}, err
}

func (client *Milvusclient) DropCollection(collectionName string) (Status, error) {
	grpcCollectionName := pb.CollectionName{collectionName, struct{}{}, nil, 0}
	grpcStatus, err := client.Instance.DropCollection(grpcCollectionName)
	if err != nil {
		return nil, err
	}
	errorCode := int64(grpcStatus.ErrorCode)
	return status{errorCode, grpcStatus.Reason}, err
}

func (client *Milvusclient) CreateIndex(indexParam *IndexParam) (Status, error) {
	keyValuePair := make([]*pb.KeyValuePair, 1)
	pair := pb.KeyValuePair{"params", indexParam.ExtraParams, struct{}{}, nil, 0}
	keyValuePair[0] = &pair
	grpcIndexParam := pb.IndexParam{nil, indexParam.CollectionName, int32(indexParam.IndexType), keyValuePair,
		struct{}{}, nil, 0}
	grpcStatus, err := client.Instance.CreateIndex(grpcIndexParam)
	if err != nil {
		return nil, err
	}
	return status{int64(grpcStatus.ErrorCode), grpcStatus.Reason}, err
}

////////////////////////////////////////////////////////////////////////////

func (client *Milvusclient) Insert(insertParam *InsertParam) ([]int64, Status, error) {
	var i int
	var rowRecordArray = make([]*pb.RowRecord, len(insertParam.RecordArray))
	for i = 0; i < len(insertParam.RecordArray); i++ {
		rowRecord := pb.RowRecord{insertParam.RecordArray[i].FloatData, insertParam.RecordArray[i].BinaryData, struct{}{}, nil, 0}
		rowRecordArray[i] = &rowRecord
	}

	grpcInsertParam := pb.InsertParam{insertParam.CollectionName, rowRecordArray, insertParam.IDArray,
		insertParam.PartitionTag, nil, struct{}{}, nil, 0}
	vectorIds, err := client.Instance.Insert(grpcInsertParam)
	if err != nil {
		return nil, nil, err
	}
	if insertParam.IDArray != nil {
		return nil, status{int64(vectorIds.Status.ErrorCode), vectorIds.Status.Reason}, err
	}
	id_array := vectorIds.VectorIdArray
	return id_array, status{int64(vectorIds.Status.ErrorCode), vectorIds.Status.Reason}, err
}

////////////////////////////////////////////////////////////////////////////

func (client *Milvusclient) GetEntityByID(collectionName string, vector_id int64) (Entity, Status, error) {
	grpcIdentity := pb.VectorIdentity{collectionName, vector_id, struct{}{}, nil, 0}
	grpcVectorData, err := client.Instance.GetVectorByID(grpcIdentity)
	if err != nil {
		return Entity{nil, nil}, nil, err
	}
	if grpcVectorData.VectorData != nil {
		return Entity{grpcVectorData.VectorData.FloatData, grpcVectorData.VectorData.BinaryData},
			status{int64(grpcVectorData.Status.ErrorCode), grpcVectorData.Status.Reason}, err
	}
	return Entity{nil, nil}, status{int64(grpcVectorData.Status.ErrorCode), grpcVectorData.Status.Reason}, err
}

////////////////////////////////////////////////////////////////////////////

func (client *Milvusclient) GetEntityIDs(getEntityIDsParam GetEntityIDsParam) ([]int64, Status, error) {
	grpcParam := pb.GetVectorIDsParam{getEntityIDsParam.CollectionName, getEntityIDsParam.SegmentName, struct{}{}, nil, 0}
	vectorIDs, err := client.Instance.GetVectorIDs(grpcParam)
	if err != nil {
		return nil, nil, err
	}
	return vectorIDs.VectorIdArray, status{int64(vectorIDs.Status.ErrorCode), vectorIDs.Status.Reason}, err
}

////////////////////////////////////////////////////////////////////////////

func (client *Milvusclient) Search(searchParam SearchParam) (TopkQueryResult, Status, error) {
	var queryRecordArray = make([]*pb.RowRecord, len(searchParam.QueryEntities))
	var i, j int64
	for i = 0; i < int64(len(searchParam.QueryEntities)); i++ {
		rowRecord := pb.RowRecord{searchParam.QueryEntities[i].FloatData, searchParam.QueryEntities[i].BinaryData, struct{}{}, nil, 0}
		queryRecordArray[i] = &rowRecord
	}

	keyValuePair := make([]*pb.KeyValuePair, 1)
	pair := pb.KeyValuePair{"params", searchParam.ExtraParams, struct{}{}, nil, 0}
	keyValuePair[0] = &pair

	grpcSearchParam := pb.SearchParam{searchParam.CollectionName, searchParam.PartitionTag, queryRecordArray,
		searchParam.Topk, keyValuePair, struct{}{}, nil, 0}
	topkQueryResult, err := client.Instance.Search(grpcSearchParam)
	if err != nil {
		return TopkQueryResult{nil}, nil, err
	}
	nq := topkQueryResult.GetRowNum()
	if nq == 0 {
		return TopkQueryResult{nil}, status{int64(topkQueryResult.Status.ErrorCode), topkQueryResult.Status.Reason,}, err
	}
	var result = make([]QueryResult, nq)
	topk := int64(len(topkQueryResult.GetIds())) / nq
	for i = 0; i < nq; i++ {
		result[i].Ids = make([]int64, topk)
		result[i].Distances = make([]float32, topk)
		for j = 0; j < topk; j++ {
			result[i].Ids[j] = topkQueryResult.GetIds()[i*topk+j]
			result[i].Distances[j] = topkQueryResult.GetDistances()[i*topk+j]
		}
	}
	return TopkQueryResult{result}, status{int64(topkQueryResult.Status.ErrorCode), topkQueryResult.Status.Reason}, nil
}

////////////////////////////////////////////////////////////////////////////

func (client *Milvusclient) DeleteByID(collectionName string, id_array []int64) (Status, error) {
	grpcParam := pb.DeleteByIDParam{collectionName, id_array, struct{}{}, nil, 0}
	grpcStatus, err := client.Instance.DeleteByID(grpcParam)
	if err != nil {
		return nil, err
	}
	return status{int64(grpcStatus.GetErrorCode()), grpcStatus.GetReason()}, err
}

////////////////////////////////////////////////////////////////////////////

func (client *Milvusclient) DescribeCollection(collectionName string) (CollectionParam, Status, error) {
	grpcCollectionName := pb.CollectionName{collectionName, struct{}{}, nil, 0}
	collectionSchema, err := client.Instance.DescribeCollection(grpcCollectionName)
	if err != nil {
		return CollectionParam{"", 0, 0, 0}, nil, err
	}
	return CollectionParam{collectionSchema.GetCollectionName(), collectionSchema.GetDimension(), collectionSchema.GetIndexFileSize(), int64(collectionSchema.GetMetricType())},
		status{int64(collectionSchema.GetStatus().GetErrorCode()), collectionSchema.Status.Reason}, err
}

////////////////////////////////////////////////////////////////////////////

func (client *Milvusclient) CountCollection(collectionName string) (int64, Status, error) {
	grpcCollectionName := pb.CollectionName{collectionName, struct{}{}, nil, 0}
	rowCount, err := client.Instance.CountCollection(grpcCollectionName)
	if err != nil {
		return 0, nil, err
	}
	return rowCount.GetCollectionRowCount(), status{int64(rowCount.GetStatus().GetErrorCode()),
		rowCount.GetStatus().GetReason()}, err
}

////////////////////////////////////////////////////////////////////////////

func (client *Milvusclient) ShowCollections() ([]string, Status, error) {
	collectionNameList, err := client.Instance.ShowCollections()
	if err != nil {
		return nil, nil, err
	}
	return collectionNameList.GetCollectionNames(), status{int64(collectionNameList.GetStatus().GetErrorCode()), collectionNameList.GetStatus().GetReason()}, err
}

////////////////////////////////////////////////////////////////////////////

func (client *Milvusclient) ShowCollectionInfo(collectionName string) (CollectionInfo, Status, error) {
	grpcCollectionName := pb.CollectionName{collectionName, struct{}{}, nil, 0}
	grpcCollectionInfo, err := client.Instance.ShowCollectionInfos(grpcCollectionName)
	if err != nil {
		return CollectionInfo{0, nil}, nil, err
	}
	partitionStats := make([]PartitionStat, len(grpcCollectionInfo.PartitionsStat))
	var i, j int
	for i = 0; i < len(grpcCollectionInfo.PartitionsStat); i++ {
		partitionStats[i].Tag = grpcCollectionInfo.PartitionsStat[i].Tag
		partitionStats[i].RowCount = grpcCollectionInfo.PartitionsStat[i].TotalRowCount
		segmentStat := make([]SegmentStat, len(grpcCollectionInfo.PartitionsStat[i].SegmentsStat))
		for j = 0; j < len(grpcCollectionInfo.GetPartitionsStat()[i].GetSegmentsStat()); j++ {
			segmentStat[j] = SegmentStat{grpcCollectionInfo.PartitionsStat[i].SegmentsStat[j].SegmentName,
				grpcCollectionInfo.PartitionsStat[i].SegmentsStat[j].RowCount,
				grpcCollectionInfo.PartitionsStat[i].SegmentsStat[j].IndexName,
				grpcCollectionInfo.PartitionsStat[i].SegmentsStat[j].DataSize}
		}
		partitionStats[i].SegmentsStat = segmentStat
	}
	collectionInfo := CollectionInfo{grpcCollectionInfo.TotalRowCount, partitionStats}
	return collectionInfo, status{int64(grpcCollectionInfo.Status.ErrorCode), grpcCollectionInfo.Status.Reason}, err
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

func (client *Milvusclient) PreloadCollection(collectionName string) (Status, error) {
	grpcCollectionName := pb.CollectionName{collectionName, struct{}{}, nil, 0}
	grpcStatus, err := client.Instance.PreloadCollection(grpcCollectionName)
	if err != nil {
		return nil, err
	}
	return status{int64(grpcStatus.GetErrorCode()), grpcStatus.GetReason()}, err
}

////////////////////////////////////////////////////////////////////////////

func (client *Milvusclient) DescribeIndex(collectionName string) (IndexParam, Status, error) {
	grpcCollectionName := pb.CollectionName{collectionName, struct{}{}, nil, 0}
	indexParam, err := client.Instance.DescribeIndex(grpcCollectionName)
	if err != nil {
		return IndexParam{"", 0, ""}, nil, err
	}
	var i int
	var extraParam string
	for i = 0; i < len(indexParam.ExtraParams); i++ {
		if indexParam.ExtraParams[i].Key == "params" {
			extraParam = indexParam.ExtraParams[i].Value
		}
	}
	return IndexParam{indexParam.GetCollectionName(), IndexType(indexParam.IndexType), extraParam},
		status{int64(indexParam.GetStatus().GetErrorCode()), indexParam.GetStatus().GetReason()}, err
}

////////////////////////////////////////////////////////////////////////////

func (client *Milvusclient) DropIndex(collectionName string) (Status, error) {
	grpcCollectionName := pb.CollectionName{collectionName, struct{}{}, nil, 0}
	grpcStatus, err := client.Instance.DropIndex(grpcCollectionName)
	if err != nil {
		return nil, err
	}
	return status{int64(grpcStatus.GetErrorCode()), grpcStatus.GetReason()}, err
}

////////////////////////////////////////////////////////////////////////////

func (client *Milvusclient) CreatePartition(partitionParam PartitionParam) (Status, error) {
	grpcPartitionParam := pb.PartitionParam{partitionParam.CollectionName,
		partitionParam.PartitionTag, struct{}{}, nil, 0}
	grpcStatus, err := client.Instance.CreatePartition(grpcPartitionParam)
	if err != nil {
		return nil, err
	}
	return status{int64(grpcStatus.GetErrorCode()), grpcStatus.GetReason()}, err
}

////////////////////////////////////////////////////////////////////////////

func (client *Milvusclient) ShowPartitions(collectionName string) ([]PartitionParam, Status, error) {
	grpcCollectionName := pb.CollectionName{collectionName, struct{}{}, nil, 0}
	grpcPartitionList, err := client.Instance.ShowPartitions(grpcCollectionName)
	if err != nil {
		return nil, status{int64(RPCFailed), err.Error()}, err
	}
	var partitionList = make([]PartitionParam, len(grpcPartitionList.GetPartitionTagArray()))
	var i int
	for i = 0; i < len(grpcPartitionList.GetPartitionTagArray()); i++ {
		partitionList[i].CollectionName = grpcPartitionList.GetPartitionTagArray()[i]
	}
	return partitionList, status{int64(grpcPartitionList.GetStatus().GetErrorCode()), grpcPartitionList.GetStatus().GetReason()}, err
}

////////////////////////////////////////////////////////////////////////////

func (client *Milvusclient) DropPartition(partitionParam PartitionParam) (Status, error) {
	grpcPartitionParam := pb.PartitionParam{partitionParam.CollectionName,
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

func (client *Milvusclient) Flush(collectionNameArray []string) (Status, error) {
	grpcParam := pb.FlushParam{collectionNameArray, struct{}{}, nil, 0}
	grpcStatus, err := client.Instance.Flush(grpcParam)
	if err != nil {
		return nil, err
	}
	return status{int64(grpcStatus.GetErrorCode()), grpcStatus.GetReason()}, err
}

////////////////////////////////////////////////////////////////////////////

func (client *Milvusclient) Compact(collectionName string) (Status, error) {
	grpcParam := pb.CollectionName{collectionName, struct{}{}, nil, 0}
	grpcStatus, err := client.Instance.Compact(grpcParam)
	if err != nil {
		return nil, err
	}
	return status{int64(grpcStatus.GetErrorCode()), grpcStatus.GetReason()}, err
}
