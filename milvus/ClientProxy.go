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
	"encoding/json"
	"errors"
	"google.golang.org/grpc/metadata"
	"math"
	"reflect"
	"strconv"
	"strings"

	pb "github.com/milvus-io/milvus-sdk-go/milvus/grpc/gen"
	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
	"google.golang.org/grpc"
)

type Milvusclient struct {
	Instance MilvusGrpcClient
}

// NewMilvusClient is the constructor of MilvusClient
func NewMilvusClient(ctx context.Context, connectParam ConnectParam) (MilvusClient, error) {
	var grpcClient MilvusGrpcClient
	client := &Milvusclient{grpcClient}
	err := client.Connect(ctx, connectParam)
	return client, err
}

func (client *Milvusclient) GetClientVersion(ctx context.Context) string {
	return clientVersion
}

func (client *Milvusclient) Connect(ctx context.Context, connectParam ConnectParam) error {
	var interceptor grpc.UnaryClientInterceptor
	interceptor = func(ctx context.Context, method string, req, reply interface{}, cc *grpc.ClientConn, invoker grpc.UnaryInvoker, opts ...grpc.CallOption) error {
		reqMD := metadata.New(map[string]string{"client_tag": connectParam.ClientTag})
		ctx = metadata.NewOutgoingContext(ctx, reqMD)
		err := invoker(ctx, method, req, reply, cc, opts...)
		return err
	}
	var opts []grpc.DialOption
	opts = append(opts, grpc.WithInsecure())
	opts = append(opts, grpc.WithBlock())
	opts = append(opts, grpc.WithDefaultCallOptions(grpc.MaxCallSendMsgSize(math.MaxInt64)))
	opts = append(opts, grpc.WithDefaultCallOptions(grpc.MaxCallRecvMsgSize(math.MaxInt64)))
	opts = append(opts, grpc.WithUnaryInterceptor(interceptor))

	serverAddr := connectParam.IPAddress + ":" + strconv.Itoa(int(connectParam.Port))

	conn, err := grpc.DialContext(ctx, serverAddr, opts...)
	if err != nil {
		return err
	}

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
	if serverVersion[0:4] != "0.11" {
		println("Server version check failed, this client supposed to connect milvus-0.11.x")
		client.Instance = nil
		err = errors.New("Connect to server failed, please check server version.")
		return err
	}

	return nil
}

func (client *Milvusclient) IsConnected(_ context.Context) bool {
	return client.Instance != nil
}

func (client *Milvusclient) Disconnect(_ context.Context) error {
	client.Instance = nil
	return nil
}

func (client *Milvusclient) CreateCollection(ctx context.Context, mapping Mapping) (Status, error) {
	fieldSize := len(mapping.Fields)
	grpcFields := make([]*pb.FieldParam, fieldSize)
	var i int
	for i = 0; i < fieldSize; i++ {
		field := mapping.Fields[i]
		grpcPair := make([]*pb.KeyValuePair, 1)
		pair := pb.KeyValuePair{"params", field.ExtraParams.params,
			struct{}{}, nil, 0}
		grpcPair[0] = &pair
		grpcFields[i] = &pb.FieldParam{0, field.Name, pb.DataType(field.Type), nil,
			grpcPair, struct{}{}, nil, 0,
		}
	}
	grpcParam := make([]*pb.KeyValuePair, 1)
	grpcParam[0] = &pb.KeyValuePair{"params", mapping.ExtraParams.params,
		struct{}{}, nil, 0}
	grpcMapping := pb.Mapping{nil, mapping.CollectionName, grpcFields, grpcParam,
		struct{}{}, nil, 0,
	}

	grpcStatus, err := client.Instance.CreateCollection(ctx, grpcMapping)
	if err != nil {
		return nil, err
	}
	errorCode := int64(grpcStatus.ErrorCode)
	return status{errorCode, grpcStatus.Reason}, err
}

func (client *Milvusclient) HasCollection(ctx context.Context, collectionName string) (bool, Status, error) {
	grpcCollectionName := pb.CollectionName{collectionName, struct{}{}, nil, 0}
	boolReply, err := client.Instance.HasCollection(ctx, grpcCollectionName)
	if err != nil {
		return false, nil, err
	}
	return boolReply.GetBoolReply(), status{int64(boolReply.GetStatus().GetErrorCode()), boolReply.GetStatus().GetReason()}, err
}

func (client *Milvusclient) DropCollection(ctx context.Context, collectionName string) (Status, error) {
	grpcCollectionName := pb.CollectionName{collectionName, struct{}{}, nil, 0}
	grpcStatus, err := client.Instance.DropCollection(ctx, grpcCollectionName)
	if err != nil {
		return nil, err
	}
	errorCode := int64(grpcStatus.ErrorCode)
	return status{errorCode, grpcStatus.Reason}, err
}

func (client *Milvusclient) CreateIndex(ctx context.Context, indexParam *IndexParam) (Status, error) {
	paramLen := len(indexParam.IndexParams)
	grpcPair := make([]*pb.KeyValuePair, paramLen)
	offset := 0
	for k, v := range indexParam.IndexParams {
		byt, _ := json.Marshal(v)
		var value string
		if k != "params" {
			value = reflect.ValueOf(v).String()
		} else {
			value = string(byt)
		}
		grpcPair[offset] = &pb.KeyValuePair{k, value, struct{}{}, nil, 0}
		offset++
	}

	grpcIndexParam := pb.IndexParam{nil, indexParam.CollectionName, indexParam.FieldName, "",
		grpcPair, struct{}{}, nil, 0}
	grpcStatus, err := client.Instance.CreateIndex(ctx, grpcIndexParam)
	if err != nil {
		return nil, err
	}
	return status{int64(grpcStatus.ErrorCode), grpcStatus.Reason}, err
}

////////////////////////////////////////////////////////////////////////////

func (client *Milvusclient) Insert(ctx context.Context, insertParam InsertParam) ([]int64, Status, error) {
	var i int

	fieldSize := len(insertParam.Fields)
	fieldValue := make([]*pb.FieldValue, fieldSize)
	for i = 0; i < fieldSize; i++ {
		var attrRecord pb.AttrRecord
		var vectorRecord pb.VectorRecord
		field := insertParam.Fields[i]
		switch t := field.RawData.(type) {
		case []int32:
			attrRecord.Int32Value = make([]int32, len(t))
			for j, value := range t {
				attrRecord.Int32Value[j] = value
			}
		case []int64:
			attrRecord.Int64Value = make([]int64, len(t))
			for j, value := range t {
				attrRecord.Int64Value[j] = value
			}
		case []float32:
			attrRecord.FloatValue = make([]float32, len(t))
			for j, value := range t {
				attrRecord.FloatValue[j] = value
			}
		case []float64:
			attrRecord.DoubleValue = make([]float64, len(t))
			for j, value := range t {
				attrRecord.DoubleValue[j] = value
			}
		case [][]float32:
			vectorRowRecords := make([]*pb.VectorRowRecord, len(t))
			for key0, rowValue := range t {
				record := make([]float32, len(rowValue))
				for key1, value := range rowValue {
					record[key1] = value
				}
				vectorRowRecord := pb.VectorRowRecord{record, nil,
					struct{}{}, nil, 0}
				vectorRowRecords[key0] = &vectorRowRecord
			}
			vectorRecord.Records = vectorRowRecords
		case [][]byte:
			vectorRowRecords := make([]*pb.VectorRowRecord, len(t))
			for key0, rowValue := range t {
				record := make([]byte, len(rowValue))
				for key1, value := range rowValue {
					record[key1] = value
				}
				vectorRowRecord := pb.VectorRowRecord{nil, record,
					struct{}{}, nil, 0}
				vectorRowRecords[key0] = &vectorRowRecord
			}
			vectorRecord.Records = vectorRowRecords
		case interface{}:
			return nil, nil, errors.New("Field value type is wrong")
		}
		fieldValue[i] = &pb.FieldValue{insertParam.Fields[i].Name, 0, &attrRecord,
			&vectorRecord, struct{}{}, nil, 0}
	}
	grpcInsertParam := pb.InsertParam{insertParam.CollectionName, fieldValue, insertParam.IDArray,
		insertParam.PartitionTag, nil, struct{}{}, nil, 0}

	vectorIds, err := client.Instance.Insert(ctx, grpcInsertParam)
	if err != nil {
		return nil, nil, err
	}
	if vectorIds.EntityIdArray == nil {
		return nil, status{int64(vectorIds.Status.ErrorCode), vectorIds.Status.Reason}, err
	}
	id_array := vectorIds.EntityIdArray
	return id_array, status{int64(vectorIds.Status.ErrorCode), vectorIds.Status.Reason}, err
}

////////////////////////////////////////////////////////////////////////////

func (client *Milvusclient) GetEntityByID(ctx context.Context, collectionName string, fields []string, entity_id []int64) ([]Entity, Status, error) {
	grpcIdentity := pb.EntityIdentity{collectionName, entity_id, fields,
		struct{}{}, nil, 0}

	grpcFieldValue, err := client.Instance.GetEntityByID(ctx, grpcIdentity)
	if err != nil {
		return nil, nil, err
	}

	var i, j int
	rowCount := len(grpcFieldValue.ValidRow)
	fieldSize := len(grpcFieldValue.Fields)
	entities := make([]Entity, rowCount)
	off := 0
	for i = 0; i < rowCount; i++ {
		entities[i].Entity = make(map[string]interface{}, fieldSize)
		entities[i].EntityId = grpcFieldValue.Ids[i]
		if grpcFieldValue.ValidRow[i] {
			for j = 0; j < fieldSize; j++ {
				if attrRecord := grpcFieldValue.Fields[j].AttrRecord; attrRecord != nil {
					if len(attrRecord.Int32Value) > 0 {
						entities[i].Entity[grpcFieldValue.Fields[j].FieldName] = attrRecord.Int32Value[off]
					} else if len(attrRecord.Int64Value) > 0 {
						entities[i].Entity[grpcFieldValue.Fields[j].FieldName] = attrRecord.Int64Value[off]
					} else if len(attrRecord.FloatValue) > 0 {
						entities[i].Entity[grpcFieldValue.Fields[j].FieldName] = attrRecord.FloatValue[off]
					} else if len(attrRecord.DoubleValue) > 0 {
						entities[i].Entity[grpcFieldValue.Fields[j].FieldName] = attrRecord.DoubleValue[off]
					}
				} else if vectorRecord := grpcFieldValue.Fields[j].VectorRecord; vectorRecord != nil {
					if floatSize := len(vectorRecord.Records[off].FloatData); floatSize > 0 {
						entities[i].Entity[grpcFieldValue.Fields[j].FieldName] = vectorRecord.Records[off].FloatData
					} else if binSize := len(vectorRecord.Records[off].BinaryData); binSize > 0 {
						entities[i].Entity[grpcFieldValue.Fields[j].FieldName] = vectorRecord.Records[off].BinaryData
					}
				}
			}
			off++
		} else {
			entities[i].Entity = nil
		}
	}

	return entities, status{int64(grpcFieldValue.Status.ErrorCode), grpcFieldValue.Status.Reason}, err
}

////////////////////////////////////////////////////////////////////////////

func (client *Milvusclient) ListIDInSegment(ctx context.Context, listIDInSegmentParam ListIDInSegmentParam) ([]int64, Status, error) {
	grpcParam := pb.GetEntityIDsParam{listIDInSegmentParam.CollectionName, listIDInSegmentParam.SegmentId,
		struct{}{}, nil, 0}
	vectorIDs, err := client.Instance.GetEntityIDs(ctx, grpcParam)
	if err != nil {
		return nil, nil, err
	}
	return vectorIDs.EntityIdArray, status{int64(vectorIDs.Status.ErrorCode), vectorIDs.Status.Reason}, err
}

////////////////////////////////////////////////////////////////////////////

func ParseDsl(dslString *string, vectorParam *string, vectorRecord *pb.VectorRecord) (string, error) {
	dsl := gjson.Parse(*dslString)
	if vectorJson := dsl.Get("vector"); vectorJson.Exists() {
		vectorDsl := dsl.Get("vector")
		var fieldName string
		vectorDsl.ForEach(func(key, value gjson.Result) bool {
			fieldName = key.String()
			return true
		})
		queryKey := fieldName + ".query"

		var floatRecords [][]float32
		_ = json.Unmarshal([]byte(vectorDsl.Get(queryKey).String()), &floatRecords)
		var byteRecords [][]byte
		if floatRecords != nil {
			vectorRecord.Records = make([]*pb.VectorRowRecord, len(floatRecords))
			for i := 0; i < len(floatRecords); i++ {
				vectorRowRecord := pb.VectorRowRecord{floatRecords[i], nil,
					struct{}{}, nil, 0}
				vectorRecord.Records[i] = &vectorRowRecord
			}
		} else if byteRecords != nil {
			vectorRecord.Records = make([]*pb.VectorRowRecord, len(byteRecords))
			for i := 0; i < len(byteRecords); i++ {
				vectorRowRecord := pb.VectorRowRecord{nil, byteRecords[i],
					struct{}{}, nil, 0}
				vectorRecord.Records[i] = &vectorRowRecord
			}
		}

		//if queryVector
		vectorParamValue, _ := sjson.Delete(vectorJson.String(), queryKey)
		valueJson := gjson.Parse(vectorParamValue)
		vectorParamKey := "placeholder_1"
		*vectorParam, _ = sjson.Set("", vectorParamKey, valueJson.Value())
		return vectorDsl.String(), nil
	}

	for _, subObject := range dsl.Array() {
		tmpObject := subObject.String()
		vectorDsl, _ := ParseDsl(&tmpObject, vectorParam, vectorRecord)
		if vectorDsl != "" {
			return vectorDsl, nil
		}
	}
	return "", nil
}

////////////////////////////////////////////////////////////////////////////

func (client *Milvusclient) Search(ctx context.Context, searchParam SearchParam) (TopkQueryResult, Status, error) {
	var vectorParam string
	var grpcVectorRecord pb.VectorRecord
	jsonDsl, err := json.Marshal(searchParam.Dsl)
	dsl := gjson.Parse(string(jsonDsl))
	boolDsl := dsl.Get("bool.must")
	boolDslString := boolDsl.String()
	vectorDsl, err := ParseDsl(&boolDslString, &vectorParam, &grpcVectorRecord)
	dslString := strings.Replace(string(jsonDsl), vectorDsl, "\"placeholder_1\"", -1)

	grpcVectorParam := pb.VectorParam{vectorParam, &grpcVectorRecord,
		struct{}{}, nil, 0}

	grpcVectorParams := make([]*pb.VectorParam, 1)
	grpcVectorParams[0] = &grpcVectorParam

	grpcSearchParam := pb.SearchParam{
		CollectionName:       searchParam.CollectionName,
		PartitionTagArray:    searchParam.PartitionTag,
		VectorParam:          grpcVectorParams,
		Dsl:                  dslString,
		ExtraParams:          nil,
		XXX_NoUnkeyedLiteral: struct{}{},
		XXX_unrecognized:     nil,
		XXX_sizecache:        0,
	}

	grpcQueryResult, err := client.Instance.Search(ctx, grpcSearchParam)
	if err != nil {
		return TopkQueryResult{nil}, nil, err
	}
	nq := grpcQueryResult.GetRowNum()
	if nq == 0 {
		return TopkQueryResult{nil}, status{int64(grpcQueryResult.Status.ErrorCode), grpcQueryResult.Status.Reason}, err
	}

	grpcFieldValue := grpcQueryResult.Entities

	var i, j int64
	rowCount := len(grpcFieldValue.ValidRow)
	fieldSize := len(grpcFieldValue.Fields)
	entities := make([]Entity, rowCount)
	off := 0
	for i = 0; i < int64(rowCount); i++ {
		entities[i].Entity = make(map[string]interface{}, fieldSize)
		for j = 0; j < int64(fieldSize); j++ {
			if grpcFieldValue.ValidRow[i] {
				if attrRecord := grpcFieldValue.Fields[j].AttrRecord; attrRecord != nil {
					if len(attrRecord.Int32Value) > 0 {
						entities[i].Entity[grpcFieldValue.Fields[j].FieldName] = attrRecord.Int32Value[off]
					} else if len(attrRecord.Int64Value) > 0 {
						entities[i].Entity[grpcFieldValue.Fields[j].FieldName] = attrRecord.Int64Value[off]
					} else if len(attrRecord.FloatValue) > 0 {
						entities[i].Entity[grpcFieldValue.Fields[j].FieldName] = attrRecord.FloatValue[off]
					} else if len(attrRecord.DoubleValue) > 0 {
						entities[i].Entity[grpcFieldValue.Fields[j].FieldName] = attrRecord.DoubleValue[off]
					}
				} else if vectorRecord := grpcFieldValue.Fields[j].VectorRecord; vectorRecord != nil {
					if floatSize := len(vectorRecord.Records[off].FloatData); floatSize > 0 {
						entities[i].Entity[grpcFieldValue.Fields[j].FieldName] = vectorRecord.Records[off].FloatData
					} else if binSize := len(vectorRecord.Records[off].BinaryData); binSize > 0 {
						entities[i].Entity[grpcFieldValue.Fields[j].FieldName] = vectorRecord.Records[off].BinaryData
					}
				}
			} else {
				entities[i].Entity[grpcFieldValue.Fields[j].FieldName] = nil
			}
		}
		if grpcFieldValue.ValidRow[i] {
			off++
		}
	}

	topkQueryResult := make([]QueryResult, nq)
	topk := int64(len(grpcQueryResult.GetDistances())) / nq
	offset := 0
	for i = 0; i < nq; i++ {
		var oneResult QueryResult
		var newSize int64
		for j = 0; j < topk; j++ {
			if grpcFieldValue.Ids[i*topk+j] != -1 {
				oneResult.Ids = append(oneResult.Ids, grpcFieldValue.Ids[i*topk+j])
				oneResult.Distances = append(oneResult.Distances, grpcQueryResult.Distances[i*topk+j])
			} else {
				newSize = j
				break
			}
		}
		if newSize == 0 {
			newSize = topk
		}
		oneResult.Entities = make([]Entity, newSize)
		oneResult.Entities = entities[offset : int64(offset)+newSize]
		offset += int(newSize)
		topkQueryResult[i] = oneResult
	}

	return TopkQueryResult{topkQueryResult}, status{int64(grpcQueryResult.Status.ErrorCode), grpcQueryResult.Status.Reason}, nil
}

////////////////////////////////////////////////////////////////////////////

func (client *Milvusclient) DeleteEntityByID(ctx context.Context, collectionName string, id_array []int64) (Status, error) {
	grpcParam := pb.DeleteByIDParam{collectionName, id_array, struct{}{}, nil, 0}
	grpcStatus, err := client.Instance.DeleteByID(ctx, grpcParam)
	if err != nil {
		return nil, err
	}
	return status{int64(grpcStatus.GetErrorCode()), grpcStatus.GetReason()}, err
}

////////////////////////////////////////////////////////////////////////////

func (client *Milvusclient) GetCollectionInfo(ctx context.Context, collectionName string) (Mapping, Status, error) {
	grpcCollectionName := pb.CollectionName{collectionName, struct{}{},
		nil, 0}
	grpcMapping, err := client.Instance.DescribeCollection(ctx, grpcCollectionName)
	if err != nil {
		return Mapping{"", nil, NewParams("")}, nil, err
	}
	fieldSize := len(grpcMapping.Fields)
	fields := make([]Field, fieldSize)
	var i, j int
	for i = 0; i < fieldSize; i++ {
		grpcField := grpcMapping.Fields[i]
		indexParam := make(map[string]string, len(grpcField.IndexParams))
		for j = 0; j < len(grpcField.IndexParams); j++ {
			indexParam[grpcField.IndexParams[j].Key] = grpcField.IndexParams[j].Value
		}
		jsonParam, _ := json.Marshal(indexParam)

		paramMap := make(map[string]interface{}, len(grpcField.ExtraParams))
		for j = 0; j < len(grpcField.ExtraParams); j++ {
			paramMap[grpcField.ExtraParams[j].Key] = grpcField.ExtraParams[j].Value
		}
		jsonExtraParam, _ := json.Marshal(paramMap)
		extraParam := string(jsonExtraParam)

		fields[i] = Field{
			Name:        grpcField.Name,
			Type:        DataType(grpcField.Type),
			IndexParams: NewParams(string(jsonParam)),
			ExtraParams: NewParams(extraParam),
		}
	}

	paramMap := make(map[string]interface{}, len(grpcMapping.ExtraParams))
	for i = 0; i < len(grpcMapping.ExtraParams); i++ {
		paramMap[grpcMapping.ExtraParams[i].Key] = grpcMapping.ExtraParams[i].Value
	}
	jsonParam, _ := json.Marshal(paramMap)

	return Mapping{grpcMapping.CollectionName, fields, NewParams(string(jsonParam))},
		status{int64(grpcMapping.GetStatus().GetErrorCode()), grpcMapping.Status.Reason}, err
}

////////////////////////////////////////////////////////////////////////////

func (client *Milvusclient) CountEntities(ctx context.Context, collectionName string) (int64, Status, error) {
	grpcCollectionName := pb.CollectionName{collectionName,
		struct{}{}, nil, 0}
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
	return collectionNameList.GetCollectionNames(), status{int64(collectionNameList.GetStatus().GetErrorCode()),
		collectionNameList.GetStatus().GetReason()}, err
}

////////////////////////////////////////////////////////////////////////////

func (client *Milvusclient) GetCollectionStats(ctx context.Context, collectionName string) (string, Status, error) {
	grpcCollectionName := pb.CollectionName{collectionName,
		struct{}{}, nil, 0}
	grpcCollectionInfo, err := client.Instance.ShowCollectionInfo(ctx, grpcCollectionName)
	if err != nil {
		return "", nil, err
	}
	return grpcCollectionInfo.JsonInfo, status{int64(grpcCollectionInfo.Status.ErrorCode),
		grpcCollectionInfo.Status.Reason}, err
}

////////////////////////////////////////////////////////////////////////////

func (client *Milvusclient) ServerVersion(ctx context.Context) (string, Status, error) {
	command := pb.Command{"version", struct{}{}, nil, 0}
	serverVersion, err := client.Instance.Cmd(ctx, command)
	if err != nil {
		return "", nil, err
	}
	return serverVersion.GetStringReply(), status{int64(serverVersion.GetStatus().GetErrorCode()),
		serverVersion.GetStatus().GetReason()}, err
}

////////////////////////////////////////////////////////////////////////////

func (client *Milvusclient) ServerStatus(ctx context.Context) (string, Status, error) {
	if client.Instance == nil {
		return "not connect to server", status{int64(0), ""}, nil
	}
	command := pb.Command{"", struct{}{}, nil, 0}
	serverStatus, err := client.Instance.Cmd(ctx, command)
	if err != nil {
		return "connection lost", nil, err
	}
	return "server alive", status{int64(serverStatus.GetStatus().GetErrorCode()),
		serverStatus.GetStatus().GetReason()}, err
}

////////////////////////////////////////////////////////////////////////////

func (client *Milvusclient) LoadCollection(ctx context.Context, collectionName string) (Status, error) {
	grpcCollectionName := pb.CollectionName{collectionName,
		struct{}{}, nil, 0}
	grpcStatus, err := client.Instance.PreloadCollection(ctx, grpcCollectionName)
	if err != nil {
		return nil, err
	}
	return status{int64(grpcStatus.GetErrorCode()), grpcStatus.GetReason()}, err
}

////////////////////////////////////////////////////////////////////////////

func (client *Milvusclient) DropIndex(ctx context.Context, collectionName string, fieldName string) (Status, error) {
	grpcIndexParam := pb.IndexParam{nil, collectionName, fieldName,
		"", nil, struct{}{}, nil, 0}
	grpcStatus, err := client.Instance.DropIndex(ctx, grpcIndexParam)
	if err != nil {
		return nil, err
	}
	return status{int64(grpcStatus.GetErrorCode()), grpcStatus.GetReason()}, err
}

////////////////////////////////////////////////////////////////////////////

func (client *Milvusclient) CreatePartition(ctx context.Context, partitionParam PartitionParam) (Status, error) {
	grpcPartitionParam := pb.PartitionParam{partitionParam.CollectionName,
		partitionParam.PartitionTag, struct{}{}, nil, 0}
	grpcStatus, err := client.Instance.CreatePartition(ctx, grpcPartitionParam)
	if err != nil {
		return nil, err
	}
	return status{int64(grpcStatus.GetErrorCode()), grpcStatus.GetReason()}, err
}

////////////////////////////////////////////////////////////////////////////

func (client *Milvusclient) ListPartitions(ctx context.Context, collectionName string) ([]PartitionParam, Status, error) {
	grpcCollectionName := pb.CollectionName{collectionName,
		struct{}{}, nil, 0}
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
	return partitionList, status{int64(grpcPartitionList.GetStatus().GetErrorCode()),
		grpcPartitionList.GetStatus().GetReason()}, err
}

////////////////////////////////////////////////////////////////////////////

func (client *Milvusclient) HasPartition(ctx context.Context, param PartitionParam) (bool, Status, error) {
	grpcParam := pb.PartitionParam{param.CollectionName, param.PartitionTag,
		struct{}{}, nil, 0}
	boolReply, err := client.Instance.HasPartition(ctx, grpcParam)
	if err != nil {
		return false, status{int64(RPCFailed), err.Error()}, err
	}
	return boolReply.BoolReply, status{int64(boolReply.Status.ErrorCode), boolReply.Status.Reason}, err
}

////////////////////////////////////////////////////////////////////////////

func (client *Milvusclient) DropPartition(ctx context.Context, partitionParam PartitionParam) (Status, error) {
	grpcPartitionParam := pb.PartitionParam{partitionParam.CollectionName,
		partitionParam.PartitionTag, struct{}{}, nil, 0}
	grpcStatus, err := client.Instance.DropPartition(ctx, grpcPartitionParam)
	if err != nil {
		return nil, err
	}
	return status{int64(grpcStatus.GetErrorCode()), grpcStatus.Reason}, err
}

////////////////////////////////////////////////////////////////////////////

func (client *Milvusclient) GetConfig(ctx context.Context, nodeName string) (string, Status, error) {
	command := pb.Command{"get_config " + nodeName,
		struct{}{}, nil, 0}
	configInfo, err := client.Instance.Cmd(ctx, command)
	if err != nil {
		return "", nil, err
	}
	return configInfo.GetStringReply(), status{int64(configInfo.GetStatus().GetErrorCode()),
		configInfo.GetStatus().GetReason()}, err
}

////////////////////////////////////////////////////////////////////////////

func (client *Milvusclient) SetConfig(ctx context.Context, nodeName string, value string) (Status, error) {
	command := pb.Command{"set_config " + nodeName + " " + value,
		struct{}{}, nil, 0}
	reply, err := client.Instance.Cmd(ctx, command)
	if err != nil {
		return nil, err
	}
	return status{int64(reply.GetStatus().GetErrorCode()), reply.GetStatus().GetReason()}, err
}

////////////////////////////////////////////////////////////////////////////

func (client *Milvusclient) Flush(ctx context.Context, collectionNameArray []string) (Status, error) {
	grpcParam := pb.FlushParam{collectionNameArray,
		struct{}{}, nil, 0}
	grpcStatus, err := client.Instance.Flush(ctx, grpcParam)
	if err != nil {
		return nil, err
	}
	return status{int64(grpcStatus.GetErrorCode()), grpcStatus.GetReason()}, err
}

////////////////////////////////////////////////////////////////////////////

func (client *Milvusclient) Compact(ctx context.Context, compact CompactParam) (Status, error) {
	grpcParam := pb.CompactParam{compact.CollectionName, compact.threshold,
		struct{}{}, nil, 0}
	grpcStatus, err := client.Instance.Compact(ctx, grpcParam)
	if err != nil {
		return nil, err
	}
	return status{int64(grpcStatus.GetErrorCode()), grpcStatus.GetReason()}, err
}
