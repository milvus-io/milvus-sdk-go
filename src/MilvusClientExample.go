package main

import (
	"strconv"
	"time"
)

var tableName string = "test_go"
var dimension int64 = 128
var indexFileSize int64 = 1024
var metricType int32 = int32(L2)
var nq int64 = 100
var nprobe int64 = 64
var nb int64 = 100000
var topk int64 = 100
var nlist int32 = 16384

func example(address string, port string) {
	var grpc_client milvusClient
	var i, j int64
	client := NewMilvusClient(grpc_client.mClient)

	//Client version
	println(client.GetClientVersion())

	//test connect
	connectParam := ConnectParam{address, port,}
	status := client.Connect(connectParam)
	if !status.ok() {
		println("client: connect failed: " + status.getMessage())
	}

	if client.IsConnected() == false {
		println("client: not connected: " + status.getMessage())
		return
	}
	println("Server status: connected")

	//Get server version
	var version string
	status, version = client.ServerVersion()
	if !status.ok() {
		println("Get server version failed: " + status.getMessage())
		return
	}
	println("Server version: " + version)

	//test create table
	tableSchema := TableSchema{tableName, dimension, indexFileSize, metricType,}
	status = client.CreateTable(tableSchema)
	if !status.ok() {
		println("Create table failed: " + status.getMessage())
		return
	}
	println("Create table " + tableName + " success")

	hasTable := client.HasTable(tableName)
	if hasTable == false {
		println("Create table failed: " + status.getMessage())
		return
	} else {
		println("Table: " + tableName + " exist")
	}

	//test show tables
	var tables []string
	status, tables = client.ShowTables()
	if !status.ok() {
		println("Show tables failed: " + status.getMessage())
		return
	}
	print("ShowTables: ")
	for i = 0; i < int64(len(tables)); i++ {
		println(tables[i])
	}

	//test insert vectors
	recordArray := make([][]float32, nb)
	for i = 0; i < nb; i++ {
		recordArray[i] = make([]float32, dimension)
		for j = 0; j < dimension; j++ {
			recordArray[i][j] = float32(i % (j + 1))
		}
	}
	insertParam := InsertParam{tableName, "", recordArray, nil,}
	status = client.Insert(&insertParam)
	if !status.ok() {
		println("Insert vector failed: " + status.getMessage())
		return
	}

	time.Sleep(3 * time.Second)

	//test describe table
	status, tableSchema = client.DescribeTable(tableName)
	if !status.ok() {
		println("Create index failed: " + status.getMessage())
		return
	}
	println("TableName:" + tableSchema.TableName + "----Dimension:" + strconv.Itoa(int(tableSchema.Dimension)) +
		"----IndexFileSize:" + strconv.Itoa(int(tableSchema.IndexFileSize)))

	//Construct query vectors
	queryVectors := make([][]float32, nq)
	for i = 0; i < nq; i++ {
		queryVectors[i] = make([]float32, dimension)
		for j = 0; j < dimension; j++ {
			queryVectors[i][j] = float32(i % (j + 1))
		}
	}

	//Search without create index
	var topkQueryResult TopkQueryResult
	searchParam := SearchParam{tableName, queryVectors, nil, topk, nprobe, nil,}
	status, topkQueryResult = client.Search(searchParam)
	for i = 0; i < int64(len(topkQueryResult.QueryResultList)); i++ {
		print(topkQueryResult.QueryResultList[i].Ids[0])
		print("        ")
		println(topkQueryResult.QueryResultList[i].Distances[0])
	}

	//test CountTable
	var tableCount int64
	status, tableCount = client.CountTable(tableName)
	if !status.ok() {
		println("Get table count failed: " + status.getMessage())
		return
	}
	println("Table count:" + strconv.Itoa(int(tableCount)))

	//Create index
	indexParam := IndexParam{tableName, IVFSQ8, nlist,}
	status = client.CreateIndex(&indexParam)
	if !status.ok() {
		println("Create index failed: " + status.getMessage())
		return
	}
	println("Create index success!")

	//Describe index
	status, indexParam = client.DescribeIndex(tableName)
	if !status.ok() {
		println("Describe index failed: " + status.getMessage())
	}
	println(indexParam.TableName + "----index type:" + strconv.Itoa(int(indexParam.IndexType)))

	//Preload table
	status = client.PreloadTable(tableName)
	if !status.ok() {
		println(status.getMessage())
	}
	println("Preload table success")

	//Search with IVFSQ8 index
	status, topkQueryResult = client.Search(searchParam)
	if !status.ok() {
		println("Search vectors failed: " + status.getMessage())
	}
	for i = 0; i < int64(len(topkQueryResult.QueryResultList)); i++ {
		print(topkQueryResult.QueryResultList[i].Ids[0])
		print("        ")
		println(topkQueryResult.QueryResultList[i].Distances[0])
	}

	//Drop index
	status = client.DropIndex(tableName)
	if !status.ok() {
		println("Drop index failed: " + status.getMessage())
	}

	//Drop table
	status = client.DropTable(tableName)
	hasTable = client.HasTable(tableName)
	if !status.ok() || hasTable == true {
		println("Drop table failed: " + status.getMessage())
		return
	}

	//Disconnect
	status = client.Disconnect()
	if !status.ok() {
		println("Disconnect failed: " + status.getMessage())
	}

	//Server status
	var serverStatus string
	status, serverStatus = client.ServerStatus()
	if !status.ok() {
		println("Get server status failed: " + status.getMessage())
	}
	println("Server status: " + serverStatus)

}

func main() {
	address := "localhost"
	port := "19530"
	example(address, port)
}
