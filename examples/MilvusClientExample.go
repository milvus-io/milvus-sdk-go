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

var collectionName string = "test_go"
var dimension int64 = 128
var indexFileSize int64 = 1024
var metricType int64 = int64(milvus.L2)
var nq int64 = 100
var nprobe int64 = 64
var nb int64 = 100000
var topk int64 = 100
var nlist int64 = 16384

func example(address string, port string) {
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
	var version string
	var status milvus.Status
	version, status, err = client.ServerVersion()
	if err != nil {
		println("Cmd rpc failed: " + err.Error())
	}
	if !status.Ok() {
		println("Get server version failed: " + status.GetMessage())
		return
	}
	println("Server version: " + version)

	//test create collection
	collectionParam := milvus.CollectionParam{collectionName, dimension, indexFileSize, metricType}
	var hasCollection bool
	//hasCollection, status, err = client.HasCollection(collectionName)
	if err != nil {
		println("HasCollection rpc failed: " + err.Error())
	}
	if hasCollection == false {
		status, err = client.CreateCollection(collectionParam)
		if err != nil {
			println("CreateCollection rpc failed: " + err.Error())
			return
		}
		if !status.Ok() {
			println("Create collection failed: " + status.GetMessage())
			return
		}
		println("Create collection " + collectionName + " success")
	}

	hasCollection, status, err = client.HasCollection(collectionName)
	if err != nil {
		println("HasCollection rpc failed: " + err.Error())
		return
	}
	if hasCollection == false {
		println("Create collection failed: " + status.GetMessage())
		return
	}
	println("Collection: " + collectionName + " exist")

	println("**************************************************")

	//test show collections
	var collections []string
	collections, status, err = client.ListCollections()
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
	records := make([]milvus.Entity, nb)
	recordArray := make([][]float32, nb)
	for i = 0; i < nb; i++ {
		recordArray[i] = make([]float32, dimension)
		for j = 0; j < dimension; j++ {
			recordArray[i][j] = float32(i % (j + 1))
		}
		records[i].FloatData = recordArray[i]
	}
	insertParam := milvus.InsertParam{collectionName, "", records, nil}
	id_array, status, err := client.Insert(&insertParam)
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
	collectionParam, status, err = client.GetCollectionInfo(collectionName)
	if err != nil {
		println("DescribeCollection rpc failed: " + err.Error())
		return
	}
	if !status.Ok() {
		println("Create index failed: " + status.GetMessage())
		return
	}
	println("CollectionName:" + collectionParam.CollectionName + "----Dimension:" + strconv.Itoa(int(collectionParam.Dimension)) +
		"----IndexFileSize:" + strconv.Itoa(int(collectionParam.IndexFileSize)))

	//Construct query vectors
	queryRecords := make([]milvus.Entity, nq)
	queryVectors := make([][]float32, nq)
	for i = 0; i < nq; i++ {
		queryVectors[i] = make([]float32, dimension)
		for j = 0; j < dimension; j++ {
			queryVectors[i][j] = float32(i % (j + 1))
		}
		queryRecords[i].FloatData = queryVectors[i]
	}

	println("**************************************************")

	//Search without create index
	var topkQueryResult milvus.TopkQueryResult
	extraParams := "{\"nprobe\" : 32}"
	searchParam := milvus.SearchParam{collectionName, queryRecords, topk, nil, extraParams}
	topkQueryResult, status, err = client.Search(searchParam)
	if err != nil {
		println("Search rpc failed: " + err.Error())
	}
	println("Search without index results: ")
	for i = 0; i < 10; i++ {
		print(topkQueryResult.QueryResultList[i].Ids[0])
		print("        ")
		println(topkQueryResult.QueryResultList[i].Distances[0])
	}

	println("**************************************************")

	//test CountCollection
	var collectionCount int64
	collectionCount, status, err = client.CountEntities(collectionName)
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
	println("Start create index...")
	extraParams = "{\"nlist\" : 16384}"
	indexParam := milvus.IndexParam{collectionName, milvus.IVFFLAT, extraParams}
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

	//Describe index
	indexParam, status, err = client.GetIndexInfo(collectionName)
	if err != nil {
		println("DescribeIndex rpc failed: " + err.Error())
		return
	}
	if !status.Ok() {
		println("Describe index failed: " + status.GetMessage())
	}
	println(indexParam.CollectionName + "----index type:" + strconv.Itoa(int(indexParam.IndexType)))

	//Preload collection
	status, err = client.LoadCollection(collectionName)
	if err != nil {
		println("PreloadCollection rpc failed: " + err.Error())
		return
	}
	if !status.Ok() {
		println(status.GetMessage())
	}
	println("Preload collection success")

	println("**************************************************")

	//Search with IVFSQ8 index
	extraParams = "{\"nprobe\" : 32}"
	searchParam = milvus.SearchParam{collectionName, queryRecords, topk, nil, extraParams}
	topkQueryResult, status, err = client.Search(searchParam)
	if err != nil {
		println("Search rpc failed: " + err.Error())
		return
	}
	if !status.Ok() {
		println("Search vectors failed: " + status.GetMessage())
	}
	println("Search with index results: ")
	for i = 0; i < 10; i++ {
		print(topkQueryResult.QueryResultList[i].Ids[0])
		print("        ")
		println(topkQueryResult.QueryResultList[i].Distances[0])
	}

	println("**************************************************")

	//Drop index
	status, err = client.DropIndex(collectionName)
	if err != nil {
		println("DropIndex rpc failed: " + err.Error())
		return
	}
	if !status.Ok() {
		println("Drop index failed: " + status.GetMessage())
	}

	//Drop collection
	status, err = client.DropCollection(collectionName)
	hasCollection, status1, err := client.HasCollection(collectionName)
	if !status.Ok() || !status1.Ok() || hasCollection == true {
		println("Drop collection failed: " + status.GetMessage())
		return
	}
	println("Drop collection " + collectionName + " success!")

	//GetConfig
	var configInfo string
	configInfo, status, err = client.GetConfig("*")
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
	var serverStatus string
	serverStatus, status, err = client.ServerStatus()
	if !status.Ok() {
		println("Get server status failed: " + status.GetMessage())
	}
	println("Server status: " + serverStatus)

}

func main() {
	address := "localhost"
	port := "19530"
	example(address, port)
}
