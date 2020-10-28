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
	"fmt"
	"github.com/milvus-io/milvus-sdk-go/milvus"
	"math/rand"
	"strconv"
	"time"
)


var collectionName string = "demp_films"
var partitionTag string = "American"
var dimension int64 = 8
var segmentRowLimit = 100000
var metricType string = string(milvus.L2)
var nq int64 = 100
var nprobe int64 = 64
var nb int64 = 3
var topk int64 = 100
var nlist int64 = 16384

var client milvus.MilvusClient

func JudgeStatus(funcName string, status milvus.Status, err error) {
	if err != nil {
		println(funcName + "rpc failed: " + err.Error())
		return
	}
	if !status.Ok() {
		println(funcName + "failed: " + status.GetMessage())
		return
	}
}


// ------
// Setup:
//    First of all, you need a runing Milvus(0.11.x). By default, Milvus runs on localhost in port 19530.
//    Then, you can use pymilvus(0.3.x) to connect to the server, You can change the _HOST and _PORT accordingly.
// ------

var HOST string = "localhost"
var PORT string = "19530"

// ------
// Basic create collection:
//     You already have a Milvus instance running, and pymilvus connecting to Milvus.
//     The first thing we will do is to create a collection `demo_films`. Incase we've already had a collection
//     named `demo_films`, we drop it before we create.
// ------
func DropAllCollections() {
	collections, status, err := client.ListCollections()
	JudgeStatus("ListCollection", status, err)
	println("ShowCollections: ")
	for _, col := range collections{
		client.DropCollection(col)
	}
}

// ------
// Basic create collection:
//     For a specific field, you can provide extra infos by a dictionary with `key = "params"`. If the field
//     has a type of `FLOAT_VECTOR` and `BINARY_VECTOR`, "dim" must be provided in extra infos. Otherwise
//     you can provide customed infos like `{"unit": "minutes"}` for you own need.
//
//     In our case, the extra infos in "duration" field means the unit of "duration" field is "minutes".
//     And `auto_id` in the parameter is set to `False` so that we can provide our own unique ids.
//     For more information you can refer to the pymilvus
//     documentation (https://milvus-io.github.io/milvus-sdk-python/pythondoc/v0.3.0/index.html).
// ------
func CreateCollection() {
	fieldByt := []byte(`{"dim": 8}`)
	fields := []milvus.Field{
		{
			"duration",
			milvus.INT32,
			"",
			"",
		},
		{
			"release_year",
			milvus.INT32,
			"",
			"",
		},
		{
			"embedding",
			milvus.VECTORFLOAT,
			"",
			string(fieldByt),
		},
	}

	colByt := []byte(`{"auto_id": false, "segment_row_limit": 5000}`)
	extraParam := string(colByt)

	//test create collection
	println("********************************Test CreateCollection********************************")
	mapping := milvus.Mapping{
		CollectionName: collectionName,
		Fields:         fields,
		ExtraParams:    extraParam,
	}
	status, err := client.CreateCollection(mapping)
	JudgeStatus("CreateCollection", status, err)
}

// ------
// Basic create collection:
//     After create collection `demo_films`, we create a partition tagged "American", it means the films we
//     will be inserted are from American.
// ------
func CreatePartition() {
	partitionParam := milvus.PartitionParam{collectionName, partitionTag}
	status, err := client.CreatePartition(partitionParam)
	JudgeStatus("CreatePartition", status, err)
}

// ------
// Basic create collection:
//     You can check the collection info and partitions we've created by `get_collection_info` and
//     `list_partitions`
// ------
func CheckCollectionInfo() {
	println("--------get collection info--------")
	mapping, status, err := client.GetCollectionInfo(collectionName)
	JudgeStatus("CreateCollection", status, err)
	println("Collection name: " + collectionName)
	for _, field := range mapping.Fields {
		fmt.Printf("field name: %-20s field type: %-10s extra params: %-20s\n",
			field.Name, strconv.Itoa(int(field.Type)), field.ExtraParams)
	}
}

// ------
// Basic insert entities:
//     We have three films of The_Lord_of_the_Rings serises here with their id, duration release_year
//     and fake embeddings to be inserted. They are listed below to give you a overview of the structure.
// ------
//	The_Lord_of_the_Rings = [
//	{
//	"title": "The_Fellowship_of_the_Ring",
//	"id": 1,
//	"duration": 208,
//	"release_year": 2001,
//	"embedding": [random.random() for _ in range(8)]
//	},
//	{
//	"title": "The_Two_Towers",
//	"id": 2,
//	"duration": 226,
//	"release_year": 2002,
//	"embedding": [random.random() for _ in range(8)]
//	},
//	{
//	"title": "The_Return_of_the_King",
//	"id": 3,
//	"duration": 252,
//	"release_year": 2003,
//	"embedding": [random.random() for _ in range(8)]
//	}
//	]
//var TheLordOfTheRings map[interface{}]interface{} = map[interface{}]interface{} {
//	{
//		"title": "The_Fellowship_of_the_Ring",
//		"id": 1,
//		"duration": 208,
//		"release_year": 2001,
//		"embedding": [random.random() for _ in range(8)]
//	},
//	{
//		"title": "The_Two_Towers",
//		"id": 2,
//		"duration": 226,
//		"release_year": 2002,
//		"embedding": [random.random() for _ in range(8)]
//	},
//	{
//		"title": "The_Return_of_the_King",
//		"id": 3,
//		"duration": 252,
//		"release_year": 2003,
//		"embedding": [random.random() for _ in range(8)]
//	}
//}


// ------
// Basic insert entities:
//     To insert these films into Milvus, we have to group values from the same field together like below.
//     Then these grouped data are used to create `hybrid_entities`.
// ------
func InsertEntities() {
	durations := []int32{202, 226, 252}
	release_years := []int32{2001, 2002, 2003}
	embedding := make([][]float32, 3)
	for  i := range embedding {
		embedding[i] = make([]float32, dimension)
		for j := range embedding[i] {
			embedding[i][j] = rand.Float32()
		}
	}
	ids := []int64 {1, 2, 3}
	fieldValue := []milvus.FieldValue {
		{
			Name:    "duration",
			RawData:      durations,
		},
		{
			Name:    "release_year",
			RawData:      release_years,
		},
		{
			Name:    "embedding",
			RawData:      embedding,
		},
	}
	insertParam := milvus.InsertParam{collectionName, fieldValue, ids, partitionTag}
	id_array, status, err := client.Insert(insertParam)
	JudgeStatus("Insert", status, err)
	if len(id_array) != 3 {
		println("ERROR: return id array is null")
	}
	println("Insert success!")
}

// ------
// Basic insert entities:
//     After insert entities into collection, we need to flush collection to make sure its on disk,
//     so that we are able to retrive it.
// ------
func Flush() {
	before_flush_counts, status, err := client.CountEntities(collectionName)
	JudgeStatus("CountEntities", status, err)
	collections := []string{collectionName}
	status, err = client.Flush(collections)
	JudgeStatus("Flush", status, err)
	after_flush_counts, status, err := client.CountEntities(collectionName)
	println("\n----------flush----------")
	fmt.Printf("There are %d films in collection `%s` before flush\n", before_flush_counts, collectionName)
	fmt.Printf("There are %d films in collection `%s` after flush\n", after_flush_counts, collectionName)
}

// ------
// Basic insert entities:
//     We can get the detail of collection statistics info by `get_collection_stats`
// ------
func GetCollectionStats() {
	info, status, err := client.GetCollectionStats(collectionName)
	JudgeStatus("GetCollectionStats", status, err)
	println("\n----------get collection stats----------")
	println(info)
}

// ------
// Basic search entities:
//     Now that we have 3 films inserted into our collection, it's time to obtain them.
//     We can get films by ids, if milvus can't find entity for a given id, `None` will be returned.
//     In the case we provide below, we will only get 1 film with id=1 and the other is `None`
// ------
func GetEntityByID() {
	ids := []int64{1, 200}
	films, status, err := client.GetEntityByID(collectionName, nil, ids)
	JudgeStatus("GetEntityByID", status, err)
	println("\n----------get entity by id = 1, id = 200----------")
	for i := range films {
		for k, v := range films[i].Entity {
			print(k + ": ")
			switch t := v.(type) {
			case int32:
				print(t)
			case []float32:
				for value := range t {
					print(strconv.Itoa(value) + " ")
				}
			}
			println("")
		}
	}
}

// ------
// Basic hybrid search entities:
//      Getting films by id is not enough, we are going to get films based on vector similarities.
//      Let's say we have a film with its `embedding` and we want to find `top3` films that are most similar
//      with it by L2 distance.
//      Other than vector similarities, we also want to obtain films that:
//        `released year` term in 2002 or 2003,
//        `duration` larger than 250 minutes.
//
//      Milvus provides Query DSL(Domain Specific Language) to support structured data filtering in queries.
//      For now milvus suppots TermQuery and RangeQuery, they are structured as below.
// ------
func Search() {
	query_embedding := make([][]float64, 1)
	for i := range query_embedding {
		query_embedding[i] = make([]float64, dimension)
		for j := range query_embedding[i] {
			query_embedding[i][j] = rand.Float64()
		}
	}
	dsl := map[string]interface{}{
		"bool": map[string]interface{}{
			"must": []map[string]interface{}{
				{
					"term": map[string]interface{}{
						"release_year": []int32{2002, 2003},
					},
				},
				{
					"range": map[string]interface{}{
						"duration": map[string]interface{}{
							"GT": 250,
						},
					},
				},
				{
					"vector": map[string]interface{}{
						"embedding": map[string]interface{}{
							"topk":        3,
							"metric_type": "L2",
							"query":       query_embedding,
						},
					},
				},
			},
		},
	}
	searchParam := milvus.SearchParam{collectionName, dsl, nil}
	topkQueryResult, status, err := client.Search(searchParam)
	JudgeStatus("Search", status, err)
	for i := 0; i < 1; i++ {
		print(topkQueryResult.QueryResultList[i].Ids[0])
		print("        ")
		println(topkQueryResult.QueryResultList[i].Distances[0])
	}
}

// ------
// Basic delete:
//     Now let's see how to delete things in Milvus.
//     You can simply delete entities by their ids.
// ------
func DeleteEntities() {
	ids := []int64{1, 2}
	status, err := client.DeleteEntityByID(collectionName, ids)
	JudgeStatus("DeleteEntityByID", status, err)
	collections := []string{collectionName}
	status, err = client.Flush(collections)
	JudgeStatus("Flush", status, err)

	result, status, err := client.GetEntityByID(collectionName, nil, ids)
	countsDeletes := len(result)
	JudgeStatus("GetEntityByID", status, err)
	countsInCollection, status, err := client.CountEntities(collectionName)
	println("\n----------delete id = 1, id = 2----------")
	fmt.Printf("Get %d entities by id 1, 2\n", countsDeletes)
	fmt.Printf("There are %d entities after delete films with 1, 2\n", countsInCollection)
}

// ------
// Basic delete:
//     You can drop partitions we create, and drop the collection we create.
// ------
func DropPartition() {
	partitionParam := milvus.PartitionParam{collectionName, partitionTag}
	status, err := client.DropPartition(partitionParam)
	JudgeStatus("DropPartition", status, err)
	collections, status, err := client.ListCollections()
	for i := range collections {
		status, err = client.DropCollection(collections[i])
		JudgeStatus("DropCollection", status, err)
	}
}

func ClientTest() {
	connectParam := milvus.ConnectParam{HOST, PORT}
	var grpcClient milvus.Milvusclient
	client = milvus.NewMilvusClient(grpcClient.Instance)
	err := client.Connect(connectParam)
	if err != nil {
		println("Client connect failed: ", err.Error())
	}
	DropAllCollections()
	CreateCollection()
	CreatePartition()
	CheckCollectionInfo()
	InsertEntities()
	Flush()
	GetCollectionStats()
	GetEntityByID()
	Search()
	DeleteEntities()
	DropPartition()
}

func main() {
	ClientTest()
}


func ClientTest_dummy(address string, port string) {
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
	fields[0].Name = "int64"
	fields[0].Type = milvus.INT64
	fields[1].Name = "float"
	fields[1].Type = milvus.FLOAT
	fields[2].Name = "float_vector"
	fields[2].Type = milvus.VECTORFLOAT

	fieldByt := []byte(`{"dim": 128}`)
	fields[2].ExtraParams = string(fieldByt)

	colByt := []byte(`{"auto_id": true, "segment_row_limit": 5000}`)
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
		Name:    "int64",
		RawData:      int64Data,
	}
	fieldValue[1] = milvus.FieldValue{
		Name:    "float",
		RawData:      floatData,
	}
	fieldValue[2] = milvus.FieldValue{
		Name:    "float_vector",
		RawData:      vectorData,
	}
	insertParam := milvus.InsertParam{collectionName, nil, nil, ""}
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
		println("field name: " + field.Name + "\t field type: " + strconv.Itoa(int(field.Type)) +
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

