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
	"errors"
	"math"

	pb "github.com/milvus-io/milvus-sdk-go/milvus/grpc/gen"
	"google.golang.org/grpc"
)

type Milvusclient struct {
	Instance MilvusGrpcClient
	conn     *grpc.ClientConn
}

// NewMilvusClient is the constructor of MilvusClient
func NewMilvusClient(ctx context.Context, connectParam ConnectParam) (MilvusClient, error) {
	client := &Milvusclient{}
	err := client.Connect(ctx, connectParam)
	return client, err
}

func (client *Milvusclient) GetClientVersion(ctx context.Context) string {
	return clientVersion
}

func (client *Milvusclient) Connect(ctx context.Context, connectParam ConnectParam) error {
	var opts []grpc.DialOption
	opts = append(opts, grpc.WithInsecure())
	opts = append(opts, grpc.WithBlock())
	opts = append(opts, grpc.WithDefaultCallOptions(grpc.MaxCallSendMsgSize(math.MaxInt64)))
	opts = append(opts, grpc.WithDefaultCallOptions(grpc.MaxCallRecvMsgSize(math.MaxInt64)))

	serverAddr := connectParam.IPAddress + ":" + connectParam.Port

	conn, err := grpc.DialContext(ctx, serverAddr, opts...)
	if err != nil {
		return err
	}
	client.conn = conn

	milvusclient := pb.NewMilvusServiceClient(conn)

	milvusGrpcClient := NewMilvusGrpcClient(milvusclient)

	client.Instance = milvusGrpcClient

	serverVersion, status, err := client.ServerVersion(ctx)
	if err != nil {
		return err
	}
	if !status.Ok() {
		println("Get server version status: " + status.GetMessage())
		return err
	}
	if serverVersion[0:3] != "1.1" {
		println("Server version check failed, this client supposed to connect milvus-1.0.x")
		client.Instance = nil
		err = errors.New("Connect to server failed, please check server version.")
		return err
	}

	return nil
}

func (client *Milvusclient) IsConnected(ctx context.Context) bool {
	return client.Instance != nil
}

func (client *Milvusclient) Disconnect(ctx context.Context) error {
	client.Instance = nil
	return client.conn.Close()
}

func (client *Milvusclient) CreateCollection(ctx context.Context, collectionParam CollectionParam) (Status, error) {
	grpcCollectionSchema := pb.CollectionSchema{
		CollectionName: collectionParam.CollectionName,
		Dimension:      collectionParam.Dimension,
		IndexFileSize:  collectionParam.IndexFileSize,
		MetricType:     collectionParam.MetricType,
	}
	grpcStatus, err := client.Instance.CreateCollection(ctx, grpcCollectionSchema)
	if err != nil {
		return nil, err
	}
	errorCode := int64(grpcStatus.ErrorCode)
	return status{errorCode, grpcStatus.Reason}, err
}

func (client *Milvusclient) HasCollection(ctx context.Context, collectionName string) (bool, Status, error) {
	grpcCollectionName := pb.CollectionName{
		CollectionName: collectionName,
	}
	boolReply, err := client.Instance.HasCollection(ctx, grpcCollectionName)
	if err != nil {
		return false, nil, err
	}
	return boolReply.GetBoolReply(), status{int64(boolReply.GetStatus().GetErrorCode()), boolReply.GetStatus().GetReason()}, err
}

func (client *Milvusclient) DropCollection(ctx context.Context, collectionName string) (Status, error) {
	grpcCollectionName := pb.CollectionName{
		CollectionName: collectionName,
	}
	grpcStatus, err := client.Instance.DropCollection(ctx, grpcCollectionName)
	if err != nil {
		return nil, err
	}
	errorCode := int64(grpcStatus.ErrorCode)
	return status{errorCode, grpcStatus.Reason}, err
}

func (client *Milvusclient) CreateIndex(ctx context.Context, indexParam *IndexParam) (Status, error) {
	keyValuePair := make([]*pb.KeyValuePair, 1)
	pair := pb.KeyValuePair{
		Key:   "params",
		Value: indexParam.ExtraParams,
	}
	keyValuePair[0] = &pair
	grpcIndexParam := pb.IndexParam{
		CollectionName: indexParam.CollectionName,
		IndexType:      int32(indexParam.IndexType),
		ExtraParams:    keyValuePair,
	}
	grpcStatus, err := client.Instance.CreateIndex(ctx, grpcIndexParam)
	if err != nil {
		return nil, err
	}
	return status{int64(grpcStatus.ErrorCode), grpcStatus.Reason}, err
}

////////////////////////////////////////////////////////////////////////////

func (client *Milvusclient) Insert(ctx context.Context, insertParam *InsertParam) ([]int64, Status, error) {
	var i int
	var rowRecordArray = make([]*pb.RowRecord, len(insertParam.RecordArray))
	for i = 0; i < len(insertParam.RecordArray); i++ {
		rowRecord := pb.RowRecord{
			FloatData:  insertParam.RecordArray[i].FloatData,
			BinaryData: insertParam.RecordArray[i].BinaryData,
		}
		rowRecordArray[i] = &rowRecord
	}

	grpcInsertParam := pb.InsertParam{
		CollectionName: insertParam.CollectionName,
		RowRecordArray: rowRecordArray,
		RowIdArray:     insertParam.IDArray,
		PartitionTag:   insertParam.PartitionTag,
	}
	vectorIds, err := client.Instance.Insert(ctx, grpcInsertParam)
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

func (client *Milvusclient) GetEntityByID(ctx context.Context, collectionName string, partitionTag string, vector_id []int64) ([]Entity, Status, error) {
	grpcIdentity := pb.VectorsIdentity{
		CollectionName: collectionName,
		PartitionTag:   partitionTag,
		IdArray:        vector_id,
	}
	grpcVectorData, err := client.Instance.GetVectorsByID(ctx, grpcIdentity)
	if err != nil {
		return nil, nil, err
	}
	if grpcVectorData.VectorsData != nil {
		entityLen := len(grpcVectorData.VectorsData)
		var entity = make([]Entity, entityLen)
		var i int64
		for i = 0; i < int64(entityLen); i++ {
			entity[i].FloatData = grpcVectorData.VectorsData[i].FloatData
			entity[i].BinaryData = grpcVectorData.VectorsData[i].BinaryData
		}
		return entity, status{int64(grpcVectorData.Status.ErrorCode), grpcVectorData.Status.Reason}, err
	}
	return nil, status{int64(grpcVectorData.Status.ErrorCode), grpcVectorData.Status.Reason}, err
}

////////////////////////////////////////////////////////////////////////////

func (client *Milvusclient) ListIDInSegment(ctx context.Context, listIDInSegmentParam ListIDInSegmentParam) ([]int64, Status, error) {
	grpcParam := pb.GetVectorIDsParam{
		CollectionName: listIDInSegmentParam.CollectionName,
		SegmentName:    listIDInSegmentParam.SegmentName,
	}
	vectorIDs, err := client.Instance.GetVectorIDs(ctx, grpcParam)
	if err != nil {
		return nil, nil, err
	}
	return vectorIDs.VectorIdArray, status{int64(vectorIDs.Status.ErrorCode), vectorIDs.Status.Reason}, err
}

////////////////////////////////////////////////////////////////////////////

func (client *Milvusclient) Search(ctx context.Context, searchParam SearchParam) (TopkQueryResult, Status, error) {
	var queryRecordArray = make([]*pb.RowRecord, len(searchParam.QueryEntities))
	var i, j int64
	for i = 0; i < int64(len(searchParam.QueryEntities)); i++ {
		rowRecord := pb.RowRecord{
			FloatData:  searchParam.QueryEntities[i].FloatData,
			BinaryData: searchParam.QueryEntities[i].BinaryData,
		}
		queryRecordArray[i] = &rowRecord
	}

	keyValuePair := make([]*pb.KeyValuePair, 1)
	pair := pb.KeyValuePair{
		Key:   "params",
		Value: searchParam.ExtraParams,
	}
	keyValuePair[0] = &pair

	grpcSearchParam := pb.SearchParam{
		CollectionName:    searchParam.CollectionName,
		PartitionTagArray: searchParam.PartitionTag,
		QueryRecordArray:  queryRecordArray,
		Topk:              searchParam.Topk,
		ExtraParams:       keyValuePair,
	}
	topkQueryResult, err := client.Instance.Search(ctx, grpcSearchParam)
	if err != nil {
		return TopkQueryResult{nil}, nil, err
	}
	nq := topkQueryResult.GetRowNum()
	if nq == 0 {
		return TopkQueryResult{nil}, status{int64(topkQueryResult.Status.ErrorCode), topkQueryResult.Status.Reason}, err
	}
	var queryResult []QueryResult
	topk := int64(len(topkQueryResult.GetIds())) / nq
	for i = 0; i < nq; i++ {
		var result QueryResult
		for j = 0; j < topk; j++ {
			if (topkQueryResult.GetIds()[i*topk+j]) != -1 {
				result.Ids = append(result.Ids, topkQueryResult.GetIds()[i*topk+j])
				result.Distances = append(result.Distances, topkQueryResult.GetDistances()[i*topk+j])
			}
		}
		queryResult = append(queryResult, result)
	}
	return TopkQueryResult{queryResult}, status{int64(topkQueryResult.Status.ErrorCode), topkQueryResult.Status.Reason}, nil
}

////////////////////////////////////////////////////////////////////////////

func (client *Milvusclient) DeleteEntityByID(ctx context.Context, collectionName string, partitionTag string, id_array []int64) (Status, error) {
	grpcParam := pb.DeleteByIDParam{
		CollectionName: collectionName,
		PartitionTag:   partitionTag,
		IdArray:        id_array,
	}
	grpcStatus, err := client.Instance.DeleteByID(ctx, grpcParam)
	if err != nil {
		return nil, err
	}
	return status{int64(grpcStatus.GetErrorCode()), grpcStatus.GetReason()}, err
}

////////////////////////////////////////////////////////////////////////////

func (client *Milvusclient) GetCollectionInfo(ctx context.Context, collectionName string) (CollectionParam, Status, error) {
	grpcCollectionName := pb.CollectionName{
		CollectionName: collectionName,
	}
	collectionSchema, err := client.Instance.DescribeCollection(ctx, grpcCollectionName)
	if err != nil {
		return CollectionParam{"", 0, 0, 0}, nil, err
	}
	return CollectionParam{collectionSchema.GetCollectionName(), collectionSchema.GetDimension(), collectionSchema.GetIndexFileSize(), collectionSchema.GetMetricType()},
		status{int64(collectionSchema.GetStatus().GetErrorCode()), collectionSchema.Status.Reason}, err
}

////////////////////////////////////////////////////////////////////////////

func (client *Milvusclient) CountEntities(ctx context.Context, collectionName string) (int64, Status, error) {
	grpcCollectionName := pb.CollectionName{
		CollectionName: collectionName,
	}
	rowCount, err := client.Instance.CountCollection(ctx, grpcCollectionName)
	if err != nil {
		return 0, nil, err
	}
	return rowCount.GetCollectionRowCount(), status{int64(rowCount.GetStatus().GetErrorCode()),
		rowCount.GetStatus().GetReason()}, err
}

////////////////////////////////////////////////////////////////////////////

func (client *Milvusclient) ListCollections(ctx context.Context) ([]string, Status, error) {
	collectionNameList, err := client.Instance.ShowCollections(ctx)
	if err != nil {
		return nil, nil, err
	}
	return collectionNameList.GetCollectionNames(), status{int64(collectionNameList.GetStatus().GetErrorCode()), collectionNameList.GetStatus().GetReason()}, err
}

////////////////////////////////////////////////////////////////////////////

func (client *Milvusclient) GetCollectionStats(ctx context.Context, collectionName string) (string, Status, error) {
	grpcCollectionName := pb.CollectionName{
		CollectionName: collectionName,
	}
	grpcCollectionInfo, err := client.Instance.ShowCollectionInfo(ctx, grpcCollectionName)
	if err != nil {
		return "", nil, err
	}
	return grpcCollectionInfo.JsonInfo, status{int64(grpcCollectionInfo.Status.ErrorCode), grpcCollectionInfo.Status.Reason}, err
}

////////////////////////////////////////////////////////////////////////////

func (client *Milvusclient) ServerVersion(ctx context.Context) (string, Status, error) {
	command := pb.Command{
		Cmd: "version",
	}
	serverVersion, err := client.Instance.Cmd(ctx, command)
	if err != nil {
		return "", nil, err
	}
	return serverVersion.GetStringReply(), status{int64(serverVersion.GetStatus().GetErrorCode()), serverVersion.GetStatus().GetReason()}, err
}

////////////////////////////////////////////////////////////////////////////

func (client *Milvusclient) ServerStatus(ctx context.Context) (string, Status, error) {
	if client.Instance == nil {
		return "not connect to server", status{int64(0), ""}, nil
	}
	command := pb.Command{Cmd: ""}
	serverStatus, err := client.Instance.Cmd(ctx, command)
	if err != nil {
		return "connection lost", nil, err
	}
	return "server alive", status{int64(serverStatus.GetStatus().GetErrorCode()), serverStatus.GetStatus().GetReason()}, err
}

////////////////////////////////////////////////////////////////////////////

func (client *Milvusclient) LoadCollection(ctx context.Context, param LoadCollectionParam) (Status, error) {
	grpcParam := pb.PreloadCollectionParam{
		CollectionName:    param.CollectionName,
		PartitionTagArray: param.PartitionTagList,
	}
	grpcStatus, err := client.Instance.PreloadCollection(ctx, grpcParam)
	if err != nil {
		return nil, err
	}
	return status{int64(grpcStatus.GetErrorCode()), grpcStatus.GetReason()}, err
}

////////////////////////////////////////////////////////////////////////////

func (client *Milvusclient) ReleaseCollection(ctx context.Context, param LoadCollectionParam) (Status, error) {
	grpcParam := pb.PreloadCollectionParam{
		CollectionName:    param.CollectionName,
		PartitionTagArray: param.PartitionTagList,
	}
	grpcStatus, err := client.Instance.ReleaseCollection(ctx, grpcParam)
	if err != nil {
		return nil, err
	}
	return status{int64(grpcStatus.GetErrorCode()), grpcStatus.GetReason()}, err
}

////////////////////////////////////////////////////////////////////////////

func (client *Milvusclient) GetIndexInfo(ctx context.Context, collectionName string) (IndexParam, Status, error) {
	grpcCollectionName := pb.CollectionName{CollectionName: collectionName}
	indexParam, err := client.Instance.DescribeIndex(ctx, grpcCollectionName)
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

func (client *Milvusclient) DropIndex(ctx context.Context, collectionName string) (Status, error) {
	grpcCollectionName := pb.CollectionName{
		CollectionName: collectionName,
	}
	grpcStatus, err := client.Instance.DropIndex(ctx, grpcCollectionName)
	if err != nil {
		return nil, err
	}
	return status{int64(grpcStatus.GetErrorCode()), grpcStatus.GetReason()}, err
}

////////////////////////////////////////////////////////////////////////////

func (client *Milvusclient) CreatePartition(ctx context.Context, partitionParam PartitionParam) (Status, error) {
	grpcPartitionParam := pb.PartitionParam{
		CollectionName: partitionParam.CollectionName,
		Tag:            partitionParam.PartitionTag,
	}
	grpcStatus, err := client.Instance.CreatePartition(ctx, grpcPartitionParam)
	if err != nil {
		return nil, err
	}
	return status{int64(grpcStatus.GetErrorCode()), grpcStatus.GetReason()}, err
}

////////////////////////////////////////////////////////////////////////////

func (client *Milvusclient) ListPartitions(ctx context.Context, collectionName string) ([]PartitionParam, Status, error) {
	grpcCollectionName := pb.CollectionName{CollectionName: collectionName}
	grpcPartitionList, err := client.Instance.ShowPartitions(ctx, grpcCollectionName)
	if err != nil {
		return nil, status{int64(RPCFailed), err.Error()}, err
	}
	var partitionList = make([]PartitionParam, len(grpcPartitionList.GetPartitionTagArray()))
	var i int
	for i = 0; i < len(grpcPartitionList.GetPartitionTagArray()); i++ {
		partitionList[i].PartitionTag = grpcPartitionList.GetPartitionTagArray()[i]
		partitionList[i].CollectionName = collectionName
	}
	return partitionList, status{int64(grpcPartitionList.GetStatus().GetErrorCode()), grpcPartitionList.GetStatus().GetReason()}, err
}

////////////////////////////////////////////////////////////////////////////

func (client *Milvusclient) DropPartition(ctx context.Context, partitionParam PartitionParam) (Status, error) {
	grpcPartitionParam := pb.PartitionParam{
		CollectionName: partitionParam.CollectionName,
		Tag:            partitionParam.PartitionTag,
	}
	grpcStatus, err := client.Instance.DropPartition(ctx, grpcPartitionParam)
	if err != nil {
		return nil, err
	}
	return status{int64(grpcStatus.GetErrorCode()), grpcStatus.Reason}, err
}

////////////////////////////////////////////////////////////////////////////

func (client *Milvusclient) GetConfig(ctx context.Context, nodeName string) (string, Status, error) {
	command := pb.Command{Cmd: "get_config " + nodeName}
	configInfo, err := client.Instance.Cmd(ctx, command)
	if err != nil {
		return "", nil, err
	}
	return configInfo.GetStringReply(), status{int64(configInfo.GetStatus().GetErrorCode()), configInfo.GetStatus().GetReason()}, err
}

////////////////////////////////////////////////////////////////////////////

func (client *Milvusclient) SetConfig(ctx context.Context, nodeName string, value string) (Status, error) {
	command := pb.Command{Cmd: "set_config " + nodeName + " " + value}
	reply, err := client.Instance.Cmd(ctx, command)
	if err != nil {
		return nil, err
	}
	return status{int64(reply.GetStatus().GetErrorCode()), reply.GetStatus().GetReason()}, err
}

////////////////////////////////////////////////////////////////////////////

func (client *Milvusclient) Flush(ctx context.Context, collectionNameArray []string) (Status, error) {
	grpcParam := pb.FlushParam{CollectionNameArray: collectionNameArray}
	grpcStatus, err := client.Instance.Flush(ctx, grpcParam)
	if err != nil {
		return nil, err
	}
	return status{int64(grpcStatus.GetErrorCode()), grpcStatus.GetReason()}, err
}

////////////////////////////////////////////////////////////////////////////

func (client *Milvusclient) Compact(ctx context.Context, collectionName string) (Status, error) {
	grpcParam := pb.CollectionName{CollectionName: collectionName}
	grpcStatus, err := client.Instance.Compact(ctx, grpcParam)
	if err != nil {
		return nil, err
	}
	return status{int64(grpcStatus.GetErrorCode()), grpcStatus.GetReason()}, err
}
