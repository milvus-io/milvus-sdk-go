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

package main

import (
	"github.com/milvus-io/milvus-sdk-go/milvus"
	"strconv"
	"time"
)

var dimension int64 = 128
var indexFileSize int64 = 1024
var metricType string = string(milvus.L2)
var nq int64 = 100
var nprobe int64 = 64
var nb int64 = 100000
var topk int64 = 100
var nlist int64 = 16384

func ClientTest(address string, port string) {
	var collectionName string = "test_go"+ strconv.Itoa(12)
	var grpcClient milvus.Milvusclient
	var i, j int64
	client := milvus.NewMilvusClient(grpcClient.Instance)

	//Client version
	println("Client version: " + client.GetClientVersion())

	//test connect
	connectParam := milvus.ConnectParam{address, port}
	err := client.Connect(connectParam)
	if err != nil {
		println("client: connect failed: " + err.Error())
	}

	if client.IsConnected() == false {
		println("client: not connected: ")
		return
	}
	println("Server status: connected")

	//Get server version
	version, status, err := client.ServerVersion()
	if err != nil {
		println("Cmd rpc failed: " + err.Error())
	}
	if !status.Ok() {
		println("Get server version failed: " + status.GetMessage())
		return
	}
	println("Server version: " + version)

	fields := make([]milvus.Field, 3)
	fields[0].FieldName = "int64"
	fields[0].DataType = milvus.INT64
	fields[1].FieldName = "float"
	fields[1].DataType = milvus.FLOAT
	fields[2].FieldName = "float_vector"
	fields[2].DataType = milvus.VECTORFLOAT

	fieldByt := []byte(`{"dim": 128}`)
	fields[2].ExtraParams = string(fieldByt)

	colByt := []byte(`{"auto_id": true, "segment_row_count": 5000}`)
	extraParam := string(colByt)

	//test create collection
	println("********************************Test CreateCollection********************************")
	mapping := milvus.Mapping{
		CollectionName: collectionName,
		Fields:         fields,
		ExtraParams:    extraParam,
	}
	status, err = client.CreateCollection(mapping)
	if err != nil {
		println("CreateCollection rpc failed: " + err.Error())
		return
	}
	if !status.Ok() {
		println("Create collection failed: " + status.GetMessage())
		return
	}
	println("Create collection " + collectionName + " success")
	hasCollection, status, err := client.HasCollection(collectionName)
	if err != nil {
		println("HasCollection rpc failed: " + err.Error())
		return
	}
	if hasCollection == false {
		println("Create collection failed: " + status.GetMessage())
		return
	}
	println("Collection: " + collectionName + " exist")

	println("********************************Test ListCollections*******************************")

	//test show collections
	collections, status, err := client.ListCollections()
	if err != nil {
		println("ShowCollections rpc failed: " + err.Error())
		return
	}
	if !status.Ok() {
		println("Show collections failed: " + status.GetMessage())
		return
	}
	println("ShowCollections: ")
	for i = 0; i < int64(len(collections)); i++ {
		println(" - " + collections[i])
	}

	//test insert vectors
	println("********************************Test Insert*****************************************")
	fieldValue := make([]milvus.FieldValue, 3)
	int64Data := make([]int64, nb)
	floatData := make([]float32, nb)
	vectorData := make([][]float32, nb)
	for i = 0; i < nb; i++ {
		int64Data[i] = i
		floatData[i] = float32(i + nb)
		vectorData[i] = make([]float32, dimension)
		for j = 0; j < dimension; j++ {
			vectorData[i][j] = float32(i % (j + 1))
		}
	}
	fieldValue[0] = milvus.FieldValue{
		FieldName:    "int64",
		RawData:      int64Data,
		IdArray:      nil,
	}
	fieldValue[1] = milvus.FieldValue{
		FieldName:    "float",
		RawData:      floatData,
		IdArray:      nil,
	}
	fieldValue[2] = milvus.FieldValue{
		FieldName:    "float_vector",
		RawData:      vectorData,
		IdArray:      nil,
	}
	insertParam := milvus.InsertParam{collectionName, "", fieldValue, nil}
	id_array, status, err := client.Insert(insertParam)
	if err != nil {
		println("Insert rpc failed: " + err.Error())
		return
	}
	if !status.Ok() {
		println("Insert vector failed: " + status.GetMessage())
		return
	}
	if len(id_array) != int(nb) {
		println("ERROR: return id array is null")
	}
	println("Insert vectors success!")

	time.Sleep(3 * time.Second)

	//test describe collection
	println("********************************Test GetCollectionInfo******************************")
	getMapping, status, err := client.GetCollectionInfo(collectionName)
	println("GetCollectionInfo finish")
	if err != nil {
		println("DescribeCollection rpc failed: " + err.Error())
		return
	}
	if !status.Ok() {
		println("Create index failed: " + status.GetMessage())
		return
	}
	println("Collection name: " + getMapping.CollectionName)
	for _, field := range getMapping.Fields {
		println("field name: " + field.FieldName + "\t field type: " + strconv.Itoa(int(field.DataType)) +
			"\t field index params: " + field.IndexParams + "\t field extra params: " + field.ExtraParams)
	}
	println("Collection extra params: " + getMapping.ExtraParams)

	//Construct query vectors
	println("********************************Test Search*****************************************")

	dsl := map[string]interface{}{
		"bool": map[string]interface{}{
			"must": []map[string]interface{}{
				{
					"vector": map[string]interface{}{
						"float_vector": map[string]interface{}{
							"topk":        topk,
							"metric_type": "L2",
							"query":       vectorData[0:5],
							"params": map[string]interface{}{
								"nprobe": 10,
							},
						},
					},
				},
			},
		},
	}
	searchParam := milvus.SearchParam{collectionName, dsl, nil}


	//Search without create index
	topkQueryResult, status, err := client.Search(searchParam)
	if err != nil {
		println("Search rpc failed: " + err.Error())
	}
	println("Search without index results: ")
	for i = 0; i < 5; i++ {
		print(topkQueryResult.QueryResultList[i].Ids[0])
		print("        ")
		println(topkQueryResult.QueryResultList[i].Distances[0])
	}

	println("******************************Test CountEntities************************************")

	//test CountCollection
	collectionCount, status, err := client.CountEntities(collectionName)
	if err != nil {
		println("CountCollection rpc failed: " + err.Error())
		return
	}
	if !status.Ok() {
		println("Get collection count failed: " + status.GetMessage())
		return
	}
	println("Collection count:" + strconv.Itoa(int(collectionCount)))

	//Create index
	println("******************************Test CreateIndex**************************************")
	println("Start create index...")
	indexParams := map[string]interface{}{
		"index_type": milvus.IVFFLAT,
		"metric_type": milvus.L2,
		"params": map[string]interface{}{
			"nlist": nlist,
		},
	}
	indexParam := milvus.IndexParam{collectionName, "float_vector", indexParams}
	status, err = client.CreateIndex(&indexParam)
	if err != nil {
		println("CreateIndex rpc failed: " + err.Error())
		return
	}
	if !status.Ok() {
		println("Create index failed: " + status.GetMessage())
		return
	}
	println("Create index success!")

	//Preload collection
	println("******************************Test LoadCollection************************************")
	status, err = client.LoadCollection(collectionName)
	if err != nil {
		println("PreloadCollection rpc failed: " + err.Error())
		return
	}
	if !status.Ok() {
		println(status.GetMessage())
	}
	println("Preload collection success")

	println("**************************Test SearchWithIVFFLATIndex********************************")

	//Search with IVFFLAT index
	topkQueryResult, status, err = client.Search(searchParam)
	if err != nil {
		println("Search rpc failed: " + err.Error())
		return
	}
	if !status.Ok() {
		println("Search vectors failed: " + status.GetMessage())
	}
	println("Search with index results: ")
	for i = 0; i < 5; i++ {
		print(topkQueryResult.QueryResultList[i].Ids[0])
		print("        ")
		println(topkQueryResult.QueryResultList[i].Distances[0])
	}

	println("******************************Test DropIndex**************************************")

	//Drop index
	status, err = client.DropIndex(collectionName, "float_vector")
	if err != nil {
		println("DropIndex rpc failed: " + err.Error())
		return
	}
	if !status.Ok() {
		println("Drop index failed: " + status.GetMessage())
	}
	println("Drop float_vector index success")

	//Drop collection
	println("******************************Test DropCollection*********************************")
	status, err = client.DropCollection(collectionName)
	hasCollection, status1, err := client.HasCollection(collectionName)
	if !status.Ok() || !status1.Ok() || hasCollection == true {
		println("Drop collection failed: " + status.GetMessage())
		return
	}
	println("Drop collection " + collectionName + " success!")

	//GetConfig
	configInfo, status, err := client.GetConfig("get_milvus_config")
	if !status.Ok() {
		println("Get config failed: " + status.GetMessage())
	}
	println("config: ")
	println(configInfo)

	//Disconnect
	err = client.Disconnect()
	if err != nil {
		println("Disconnect failed!")
		return
	}
	println("Client disconnect server success!")

	//Server status
	serverStatus, status, err := client.ServerStatus()
	if !status.Ok() {
		println("Get server status failed: " + status.GetMessage())
	}
	println("Server status: " + serverStatus)

}

func main() {
	address := "localhost"
	port := "19530"
	ClientTest(address, port)
}
