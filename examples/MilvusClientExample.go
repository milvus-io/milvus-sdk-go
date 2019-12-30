package main

import (
	"github.com/milvus-io/milvus-sdk-go/milvus"
	"strconv"
	"time"
)

var tableName string = "test_go"
var dimension int64 = 128
var indexFileSize int64 = 1024
var metricType int32 = int32(milvus.L2)
var nq int64 = 100
var nprobe int64 = 64
var nb int64 = 100000
var topk int64 = 100
var nlist int32 = 16384

func example(address string, port string) {
	var grpcClient milvus.Milvusclient
	var i, j int64
	client := milvus.NewMilvusClient(grpcClient.MClient)

	//Client version
	println(client.GetClientVersion())

	//test connect
	connectParam := milvus.ConnectParam{address, port}
	status := client.Connect(connectParam)
	if !status.Ok() {
		println("client: connect failed: " + status.GetMessage())
	}

	if client.IsConnected() == false {
		println("client: not connected: " + status.GetMessage())
		return
	}
	println("Server status: connected")

	//Get server version
	var version string
	status, version = client.ServerVersion()
	if !status.Ok() {
		println("Get server version failed: " + status.GetMessage())
		return
	}
	println("Server version: " + version)

	//test create table
	tableSchema := milvus.TableSchema{tableName, dimension, indexFileSize, metricType}
	if client.HasTable(tableName) == false {
		status = client.CreateTable(tableSchema)
		if !status.Ok() {
			println("Create table failed: " + status.GetMessage())
			return
		}
		println("Create table " + tableName + " success")
	}

	hasTable := client.HasTable(tableName)
	if hasTable == false {
		println("Create table failed: " + status.GetMessage())
		return
	}
	println("Table: " + tableName + " exist")

	//test show tables
	var tables []string
	status, tables = client.ShowTables()
	if !status.Ok() {
		println("Show tables failed: " + status.GetMessage())
		return
	}
	println("ShowTables: ")
	for i = 0; i < int64(len(tables)); i++ {
		println(" - " + tables[i])
	}

	//test insert vectors
	recordArray := make([][]float32, nb)
	for i = 0; i < nb; i++ {
		recordArray[i] = make([]float32, dimension)
		for j = 0; j < dimension; j++ {
			recordArray[i][j] = float32(i % (j + 1))
		}
	}
	insertParam := milvus.InsertParam{tableName, "", recordArray, nil}
	status = client.Insert(&insertParam)
	if !status.Ok() {
		println("Insert vector failed: " + status.GetMessage())
		return
	}

	time.Sleep(3 * time.Second)

	//test describe table
	status, tableSchema = client.DescribeTable(tableName)
	if !status.Ok() {
		println("Create index failed: " + status.GetMessage())
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
	var topkQueryResult milvus.TopkQueryResult
	searchParam := milvus.SearchParam{tableName, queryVectors, nil, topk, nprobe, nil}
	status, topkQueryResult = client.Search(searchParam)
	for i = 0; i < 10; i++ {
		print(topkQueryResult.QueryResultList[i].Ids[0])
		print("        ")
		println(topkQueryResult.QueryResultList[i].Distances[0])
	}

	//test CountTable
	var tableCount int64
	status, tableCount = client.CountTable(tableName)
	if !status.Ok() {
		println("Get table count failed: " + status.GetMessage())
		return
	}
	println("Table count:" + strconv.Itoa(int(tableCount)))

	//Create index
	indexParam := milvus.IndexParam{tableName, milvus.IVFSQ8, nlist}
	status = client.CreateIndex(&indexParam)
	if !status.Ok() {
		println("Create index failed: " + status.GetMessage())
		return
	}
	println("Create index success!")

	//Describe index
	status, indexParam = client.DescribeIndex(tableName)
	if !status.Ok() {
		println("Describe index failed: " + status.GetMessage())
	}
	println(indexParam.TableName + "----index type:" + strconv.Itoa(int(indexParam.IndexType)))

	//Preload table
	status = client.PreloadTable(tableName)
	if !status.Ok() {
		println(status.GetMessage())
	}
	println("Preload table success")

	//Search with IVFSQ8 index
	status, topkQueryResult = client.Search(searchParam)
	if !status.Ok() {
		println("Search vectors failed: " + status.GetMessage())
	}
	for i = 0; i < 10; i++ {
		print(topkQueryResult.QueryResultList[i].Ids[0])
		print("        ")
		println(topkQueryResult.QueryResultList[i].Distances[0])
	}

	//Drop index
	status = client.DropIndex(tableName)
	if !status.Ok() {
		println("Drop index failed: " + status.GetMessage())
	}

	//Drop table
	status = client.DropTable(tableName)
	hasTable = client.HasTable(tableName)
	if !status.Ok() || hasTable == true {
		println("Drop table failed: " + status.GetMessage())
		return
	}

	//Disconnect
	status = client.Disconnect()
	if !status.Ok() {
		println("Disconnect failed: " + status.GetMessage())
	}

	//Server status
	var serverStatus string
	status, serverStatus = client.ServerStatus()
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
