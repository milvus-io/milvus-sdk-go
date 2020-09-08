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
	pb "github.com/milvus-io/milvus-sdk-go/milvus/grpc/gen"
	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
	"google.golang.org/grpc"
	"math"
	"reflect"
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
	opts = append(opts, grpc.WithDefaultCallOptions(grpc.MaxCallSendMsgSize(math.MaxInt64)))
	opts = append(opts, grpc.WithDefaultCallOptions(grpc.MaxCallRecvMsgSize(math.MaxInt64)))

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

	serverVersion, status, err := client.ServerVersion()
	if err != nil {
		return err
	}
	if !status.Ok() {
		println("Get server version status: " + status.GetMessage())
		return err
	}
	if serverVersion[0:5] != "0.11" {
		println("Server version check failed, this client supposed to connect milvus-0.10.x")
		client.Instance = nil
		err = errors.New("Connecto server failed, please check server version.")
		return err
	}

	return nil
}

func (client *Milvusclient) IsConnected() bool {
	return client.Instance != nil
}

func (client *Milvusclient) Disconnect() error {
	client.Instance = nil
	return nil
}

func (client *Milvusclient) CreateCollection(mapping Mapping) (Status, error) {
	fieldSize := len(mapping.Fields)
	grpcFields := make([]*pb.FieldParam, fieldSize)
	var i int
	for i = 0; i < fieldSize; i++ {
		field := mapping.Fields[i]
		grpcPair := make([]*pb.KeyValuePair, 1)
		pair := pb.KeyValuePair{"params", field.ExtraParams,
			struct{}{}, nil, 0,}
		grpcPair[0] = &pair
		grpcFields[i] = &pb.FieldParam{0, field.FieldName, pb.DataType(field.DataType), nil,
			grpcPair, struct{}{}, nil, 0,
		}
	}
	grpcParam := make([]*pb.KeyValuePair, 1)
	grpcParam[0] = &pb.KeyValuePair{"params", mapping.ExtraParams,
		struct{}{}, nil, 0,}
	grpcMapping := pb.Mapping{nil, mapping.CollectionName, grpcFields, nil,
		struct{}{}, nil, 0,
	}

	grpcStatus, err := client.Instance.CreateCollection(grpcMapping)
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
	var i int
	paramLen := len(indexParam.IndexParams)
	grpcPair := make([]*pb.KeyValuePair, paramLen)
	for k, v := range indexParam.IndexParams {
		grpcPair[i] = &pb.KeyValuePair{k, v.(string), struct{}{}, nil, 0,}
	}

	grpcIndexParam := pb.IndexParam{nil, indexParam.CollectionName, indexParam.FieldName, nil,
		grpcPair, struct{}{}, nil, 0,}
	grpcStatus, err := client.Instance.CreateIndex(grpcIndexParam)
	if err != nil {
		return nil, err
	}
	return status{int64(grpcStatus.ErrorCode), grpcStatus.Reason}, err
}

////////////////////////////////////////////////////////////////////////////

func (client *Milvusclient) Insert(insertParam InsertParam) ([]int64, Status, error) {
	var i int

	fieldSize := len(insertParam.Fields)
	fieldValue := make([]*pb.FieldValue, fieldSize)
	for i = 0; i < fieldSize; i++ {
		var attrRecord pb.AttrRecord
		var vectorRecord pb.VectorRecord
		field := insertParam.Fields[i]
		switch t := field.RawData.(type) {
		case []int32:
			for _, value := range t {
				attrRecord.Int32Value = append(attrRecord.Int32Value, value)
			}
		case []int64:
			for _, value := range t {
				attrRecord.Int64Value = append(attrRecord.Int64Value, value)
			}
		case []float32:
			for _, value := range t {
				attrRecord.FloatValue = append(attrRecord.FloatValue, value)
			}
		case []float64:
			for _, value := range t {
				attrRecord.DoubleValue = append(attrRecord.DoubleValue, value)
			}
		case [][]float32:
			vectorRowRecords := make([]*pb.VectorRowRecord, len(t))
			for _, rowValue := range t {
				record := make([]float32, len(rowValue))
				for _, value := range rowValue {
					record = append(record, value)
				}
				vectorRowRecord := pb.VectorRowRecord{record, nil,
					struct{}{}, nil, 0,}
				vectorRowRecords = append(vectorRowRecords, &vectorRowRecord)
			}
			vectorRecord.Records = vectorRowRecords
		case [][]byte:
			vectorRowRecords := make([]*pb.VectorRowRecord, len(t))
			for _, rowValue := range t {
				record := make([]byte, len(rowValue))
				for _, value := range rowValue {
					record = append(record, value)
				}
				vectorRowRecord := pb.VectorRowRecord{nil, record,
					struct{}{}, nil, 0}
				vectorRowRecords = append(vectorRowRecords, &vectorRowRecord)
			}
		case interface{}:
			return nil, nil, errors.New("Field value type is wrong")
		}
		fieldValue[i] = &pb.FieldValue{insertParam.Fields[i].FieldName, 0, &attrRecord,
			&vectorRecord, struct{}{}, nil, 0}
	}

	grpcInsertParam := pb.InsertParam{insertParam.CollectionName, fieldValue, insertParam.IDArray,
		insertParam.PartitionTag, nil, struct{}{}, nil, 0}

	vectorIds, err := client.Instance.Insert(grpcInsertParam)
	if err != nil {
		return nil, nil, err
	}
	if insertParam.IDArray != nil {
		return nil, status{int64(vectorIds.Status.ErrorCode), vectorIds.Status.Reason}, err
	}
	id_array := vectorIds.EntityIdArray
	return id_array, status{int64(vectorIds.Status.ErrorCode), vectorIds.Status.Reason}, err
}

////////////////////////////////////////////////////////////////////////////

func (client *Milvusclient) GetEntityByID(collectionName string, fieldName []string, entity_id []int64) ([]Entity, Status, error) {
	grpcIdentity := pb.EntityIdentity{collectionName, entity_id, fieldName,
		struct{}{}, nil, 0}

	grpcFieldValue, err := client.Instance.GetEntityByID(grpcIdentity)
	if err != nil {
		return nil, nil, err
	}

	var i, j int
	var validIds []int64
	for i = 0; i < len(grpcFieldValue.ValidRow); i++ {
		if grpcFieldValue.ValidRow[i] {
			validIds = append(validIds, grpcFieldValue.Ids[i])
		}
	}

	fieldSize := len(grpcFieldValue.Fields)
	entities := make([]Entity, len(validIds))
	for i = 0; i < fieldSize; i++ {
		if attrRecord := grpcFieldValue.Fields[i].AttrRecord; attrRecord != nil {
			for j = 0; j < len(validIds); j++ {
				if len(attrRecord.Int32Value) > 0 {
					entities[j].Entity[grpcFieldValue.Fields[i].FieldName] = attrRecord.Int32Value[j]
				} else if len(attrRecord.Int64Value) > 0 {
					entities[j].Entity[grpcFieldValue.Fields[i].FieldName] = attrRecord.Int64Value[j]
				} else if len(attrRecord.FloatValue) > 0 {
					entities[j].Entity[grpcFieldValue.Fields[i].FieldName] = attrRecord.FloatValue[j]
				} else if len(attrRecord.DoubleValue) > 0 {
					entities[j].Entity[grpcFieldValue.Fields[i].FieldName] = attrRecord.DoubleValue[j]
				}
			}
		} else if vectorRecord := grpcFieldValue.Fields[i].VectorRecord; vectorRecord != nil {
			for j = 0; j < len(validIds); j++ {
				if floatSize := len(vectorRecord.Records[j].FloatData); floatSize > 0 {
					floatData := make([]float32, floatSize)
					copy(floatData, vectorRecord.Records[j].FloatData)
					entities[j].Entity[grpcFieldValue.Fields[i].FieldName] = floatData
				} else if binSize := len(vectorRecord.Records[j].BinaryData); binSize > 0 {
					binData := make([]byte, binSize)
					copy(binData, vectorRecord.Records[j].BinaryData)
					entities[j].Entity[grpcFieldValue.Fields[i].FieldName] = binData
				}
			}
		}
	}

	return entities, status{int64(grpcFieldValue.Status.ErrorCode), grpcFieldValue.Status.Reason}, err
}

////////////////////////////////////////////////////////////////////////////

func (client *Milvusclient) ListIDInSegment(listIDInSegmentParam ListIDInSegmentParam) ([]int64, Status, error) {
	grpcParam := pb.GetEntityIDsParam{listIDInSegmentParam.CollectionName, listIDInSegmentParam.SegmentId,
		struct{}{}, nil, 0}
	vectorIDs, err := client.Instance.GetEntityIDs(grpcParam)
	if err != nil {
		return nil, nil, err
	}
	return vectorIDs.EntityIdArray, status{int64(vectorIDs.Status.ErrorCode), vectorIDs.Status.Reason}, err
}

////////////////////////////////////////////////////////////////////////////

func ParseDsl(dsl gjson.Result, vectorParam gjson.Result, vectorRecord *pb.VectorRecord) error {
	if vectorParam := dsl.Get("vector"); vectorParam.Exists() {
		queryValue := dsl.Get("vector.query").Value()
		object := reflect.ValueOf(queryValue)

		var items []interface{}
		for i := 0; i < object.Len(); i++ {
			items = append(items, object.Index(i).Interface())
		}

		var float_records [][]float32
		var byte_records [][]byte
		for _, v := range items {
			item := reflect.ValueOf(v)
			var float_record []float32
			var byte_record []byte
			for i := 0; i < item.NumField(); i++ {
				itm := item.Field(i).Interface()
				switch t := itm.(type) {
				case float32:
					float_record = append(float_record, t)
				case byte:
					byte_record = append(byte_record, t)
				}
			}
			float_records = append(float_records, float_record)
			byte_records = append(byte_records, byte_record)
		}
		if float_records != nil {
			vectorRecord.Records = make([]*pb.VectorRowRecord, len(float_records))
			for i := 0; i < len(float_records); i++ {
				vectorRecord.Records[i].FloatData = make([]float32, len(float_records[i]))
				copy(vectorRecord.Records[i].FloatData, float_records[i])
			}
		} else if byte_records != nil {
			vectorRecord.Records = make([]*pb.VectorRowRecord, len(byte_records))
			for i := 0; i < len(byte_records); i++ {
				vectorRecord.Records[i].BinaryData = make([]byte, len(byte_records[i]))
				copy(vectorRecord.Records[i].BinaryData, byte_records[i])
			}
		}

		//if queryVector
		sjson.Set(dsl.String(), "vector", "placeholder_1")
		sjson.Delete(vectorParam.String(), "query")
		return nil
	}

	for _, subObject := range dsl.Array() {
		_ = ParseDsl(subObject, vectorParam, vectorRecord)
	}
	return nil
}

////////////////////////////////////////////////////////////////////////////

func (client *Milvusclient) Search(searchParam SearchParam) (TopkQueryResult, Status, error) {
	var vectorParam gjson.Result
	var grpcVectorRecord pb.VectorRecord
	dsl := gjson.Parse(searchParam.Dsl)
	err := ParseDsl(dsl, vectorParam, &grpcVectorRecord)

	grpcVectorParam := pb.VectorParam{vectorParam.Str, &grpcVectorRecord,
		struct{}{}, nil, 0,}

	grpcVectorParams := make([]*pb.VectorParam, 1)
	grpcVectorParams = append(grpcVectorParams, &grpcVectorParam)

	keyValuePair := make([]*pb.KeyValuePair, 1)
	pair := pb.KeyValuePair{"params", searchParam.ExtraParams, struct{}{}, nil, 0}
	keyValuePair[0] = &pair

	grpcSearchParam := pb.SearchParam{
		CollectionName:       searchParam.CollectionName,
		PartitionTagArray:    searchParam.PartitionTag,
		VectorParam:          grpcVectorParams,
		Dsl:                  dsl.Str,
		ExtraParams:          keyValuePair,
		XXX_NoUnkeyedLiteral: struct{}{},
		XXX_unrecognized:     nil,
		XXX_sizecache:        0,
	}

	grpcQueryResult, err := client.Instance.Search(grpcSearchParam)
	if err != nil {
		return TopkQueryResult{nil}, nil, err
	}
	nq := grpcQueryResult.GetRowNum()
	if nq == 0 {
		return TopkQueryResult{nil}, status{int64(grpcQueryResult.Status.ErrorCode), grpcQueryResult.Status.Reason,}, err
	}

	grpcFieldValue := grpcQueryResult.Entities

	var i, j int64
	var validIds []int64
	for i = 0; i < int64(len(grpcFieldValue.ValidRow)); i++ {
		if grpcFieldValue.ValidRow[i] {
			validIds = append(validIds, grpcFieldValue.Ids[i])
		}
	}

	queryResult := make([]QueryResult, 0, nq)

	fieldSize := len(grpcFieldValue.Fields)
	entities := make([]Entity, len(validIds))
	for i = 0; i < int64(fieldSize); i++ {
		if attrRecord := grpcFieldValue.Fields[i].AttrRecord; attrRecord != nil {
			for j = 0; j < int64(len(validIds)); j++ {
				if len(attrRecord.Int32Value) > 0 {
					entities[j].Entity[grpcFieldValue.Fields[i].FieldName] = attrRecord.Int32Value[j]
				} else if len(attrRecord.Int64Value) > 0 {
					entities[j].Entity[grpcFieldValue.Fields[i].FieldName] = attrRecord.Int64Value[j]
				} else if len(attrRecord.FloatValue) > 0 {
					entities[j].Entity[grpcFieldValue.Fields[i].FieldName] = attrRecord.FloatValue[j]
				} else if len(attrRecord.DoubleValue) > 0 {
					entities[j].Entity[grpcFieldValue.Fields[i].FieldName] = attrRecord.DoubleValue[j]
				}
			}
		} else if vectorRecord := grpcFieldValue.Fields[i].VectorRecord; vectorRecord != nil {
			for j = 0; j < int64(len(validIds)); j++ {
				if floatSize := len(vectorRecord.Records[j].FloatData); floatSize > 0 {
					floatData := make([]float32, floatSize)
					copy(floatData, vectorRecord.Records[j].FloatData)
					entities[j].Entity[grpcFieldValue.Fields[i].FieldName] = floatData
				} else if binSize := len(vectorRecord.Records[j].BinaryData); binSize > 0 {
					binData := make([]byte, binSize)
					copy(binData, vectorRecord.Records[j].BinaryData)
					entities[j].Entity[grpcFieldValue.Fields[i].FieldName] = binData
				}
			}
		}
	}

	topkQueryResult := make([]QueryResult, nq)
	topk := int64(len(grpcQueryResult.GetDistances())) / nq
	offset := 0
	for i = 0; i < nq; i++ {
		var oneResult QueryResult
		var newSize int64
		for j = 0; j < topk; j++ {
			if grpcFieldValue.Ids[i * topk + j] != -1 {
				oneResult.Ids = append(oneResult.Ids, grpcFieldValue.Ids[i* topk +j])
				oneResult.Distances = append(oneResult.Distances, grpcQueryResult.Distances[i* topk + j])
			} else {
				newSize = j
				break
			}
		}
		oneResult.Entities = make([]Entity, newSize)
		oneResult.Entities = entities[offset : int64(offset) + newSize]
		offset += int(newSize)
		topkQueryResult = append(topkQueryResult, oneResult)
	}

	return TopkQueryResult{queryResult}, status{int64(grpcQueryResult.Status.ErrorCode), grpcQueryResult.Status.Reason}, nil
}

////////////////////////////////////////////////////////////////////////////

func (client *Milvusclient) DeleteEntityByID(collectionName string, id_array []int64) (Status, error) {
	grpcParam := pb.DeleteByIDParam{collectionName, id_array, struct{}{}, nil, 0}
	grpcStatus, err := client.Instance.DeleteByID(grpcParam)
	if err != nil {
		return nil, err
	}
	return status{int64(grpcStatus.GetErrorCode()), grpcStatus.GetReason()}, err
}

////////////////////////////////////////////////////////////////////////////

func (client *Milvusclient) GetCollectionInfo(collectionName string) (Mapping, Status, error) {
	grpcCollectionName := pb.CollectionName{collectionName, struct{}{},
		nil, 0}
	grpcMapping, err := client.Instance.DescribeCollection(grpcCollectionName)
	if err != nil {
		return Mapping{"", nil, ""}, nil, err
	}
	fieldSize := len(grpcMapping.Fields)
	fields := make([]Field, fieldSize)
	var i, j int
	for i = 0; i < fieldSize; i++ {
		grpcField := grpcMapping.Fields[i]
		var indexParam map[string]string
		for j = 0; j < len(grpcField.IndexParams); j++ {
			indexParam[grpcField.IndexParams[j].Key] = grpcField.IndexParams[j].Value
		}
		jsonParam, _ := json.Marshal(indexParam)

		var paramMap map[string]interface{}
		for j = 0; j < len(grpcField.ExtraParams); j++ {
			paramMap[grpcField.ExtraParams[j].Key] = grpcField.ExtraParams[j].Value
		}
		jsonExtraParam, _ := json.Marshal(paramMap)
		extraParam := string(jsonExtraParam)

		fields[i] = Field{
			FieldName:   grpcField.Name,
			DataType:    DataType(grpcField.Type),
			IndexParams: string(jsonParam),
			ExtraParams: extraParam,
		}
	}

	var paramMap map[string]interface{}
	for i = 0; i < len(grpcMapping.ExtraParams); i++ {
		paramMap[grpcMapping.ExtraParams[i].Key] = grpcMapping.ExtraParams[j].Value
	}
	jsonParam, _ := json.Marshal(paramMap)

	return Mapping{grpcMapping.CollectionName, fields, string(jsonParam)},
		status{int64(grpcMapping.GetStatus().GetErrorCode()), grpcMapping.Status.Reason}, err
}

////////////////////////////////////////////////////////////////////////////

func (client *Milvusclient) CountEntities(collectionName string) (int64, Status, error) {
	grpcCollectionName := pb.CollectionName{collectionName,
		struct{}{}, nil, 0}
	rowCount, err := client.Instance.CountCollection(grpcCollectionName)
	if err != nil {
		return 0, nil, err
	}
	return rowCount.GetCollectionRowCount(), status{int64(rowCount.GetStatus().GetErrorCode()),
		rowCount.GetStatus().GetReason()}, err
}

////////////////////////////////////////////////////////////////////////////

func (client *Milvusclient) ListCollections() ([]string, Status, error) {
	collectionNameList, err := client.Instance.ShowCollections()
	if err != nil {
		return nil, nil, err
	}
	return collectionNameList.GetCollectionNames(), status{int64(collectionNameList.GetStatus().GetErrorCode()),
		collectionNameList.GetStatus().GetReason()}, err
}

////////////////////////////////////////////////////////////////////////////

func (client *Milvusclient) GetCollectionStats(collectionName string) (string, Status, error) {
	grpcCollectionName := pb.CollectionName{collectionName,
		struct{}{}, nil, 0}
	grpcCollectionInfo, err := client.Instance.ShowCollectionInfo(grpcCollectionName)
	if err != nil {
		return "", nil, err
	}
	return grpcCollectionInfo.JsonInfo, status{int64(grpcCollectionInfo.Status.ErrorCode),
		grpcCollectionInfo.Status.Reason}, err
}

////////////////////////////////////////////////////////////////////////////

func (client *Milvusclient) ServerVersion() (string, Status, error) {
	command := pb.Command{"version", struct{}{}, nil, 0}
	serverVersion, err := client.Instance.Cmd(command)
	if err != nil {
		return "", nil, err
	}
	return serverVersion.GetStringReply(), status{int64(serverVersion.GetStatus().GetErrorCode()),
		serverVersion.GetStatus().GetReason()}, err
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
	return "server alive", status{int64(serverStatus.GetStatus().GetErrorCode()),
		serverStatus.GetStatus().GetReason()}, err
}

////////////////////////////////////////////////////////////////////////////

func (client *Milvusclient) LoadCollection(collectionName string) (Status, error) {
	grpcCollectionName := pb.CollectionName{collectionName,
		struct{}{}, nil, 0}
	grpcStatus, err := client.Instance.PreloadCollection(grpcCollectionName)
	if err != nil {
		return nil, err
	}
	return status{int64(grpcStatus.GetErrorCode()), grpcStatus.GetReason()}, err
}

////////////////////////////////////////////////////////////////////////////

func (client *Milvusclient) DropIndex(indexParam IndexParam) (Status, error) {
	grpcIndexParam := pb.IndexParam{nil, indexParam.CollectionName, indexParam.FieldName,
		"", nil, struct{}{}, nil, 0}
	grpcStatus, err := client.Instance.DropIndex(grpcIndexParam)
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

func (client *Milvusclient) ListPartitions(collectionName string) ([]PartitionParam, Status, error) {
	grpcCollectionName := pb.CollectionName{collectionName,
		struct{}{}, nil, 0}
	grpcPartitionList, err := client.Instance.ShowPartitions(grpcCollectionName)
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
	command := pb.Command{"get_config " + nodeName,
		struct{}{}, nil, 0}
	configInfo, err := client.Instance.Cmd(command)
	if err != nil {
		return "", nil, err
	}
	return configInfo.GetStringReply(), status{int64(configInfo.GetStatus().GetErrorCode()),
		configInfo.GetStatus().GetReason()}, err
}

////////////////////////////////////////////////////////////////////////////

func (client *Milvusclient) SetConfig(nodeName string, value string) (Status, error) {
	command := pb.Command{"set_config " + nodeName + " " + value,
		struct{}{}, nil, 0}
	reply, err := client.Instance.Cmd(command)
	if err != nil {
		return nil, err
	}
	return status{int64(reply.GetStatus().GetErrorCode()), reply.GetStatus().GetReason()}, err
}

////////////////////////////////////////////////////////////////////////////

func (client *Milvusclient) Flush(collectionNameArray []string) (Status, error) {
	grpcParam := pb.FlushParam{collectionNameArray,
		struct{}{}, nil, 0}
	grpcStatus, err := client.Instance.Flush(grpcParam)
	if err != nil {
		return nil, err
	}
	return status{int64(grpcStatus.GetErrorCode()), grpcStatus.GetReason()}, err
}

////////////////////////////////////////////////////////////////////////////

func (client *Milvusclient) Compact(compact CompactParam) (Status, error) {
	grpcParam := pb.CompactParam{compact.CollectionName, compact.threshold,
		struct{}{}, nil, 0}
	grpcStatus, err := client.Instance.Compact(grpcParam)
	if err != nil {
		return nil, err
	}
	return status{int64(grpcStatus.GetErrorCode()), grpcStatus.GetReason()}, err
}
