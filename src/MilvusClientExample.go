package main

var tableName string = "test_go"
var dimension int64  = 128
var indexFileSize int64 = 1024
var metricType int32 = int32(L2)
var nq int64 = 100
var nprobe int64 = 64
var nb int64 = 100000

func example(address string, port string) {
	var grpc_client milvusClient
	client := NewMilvusClient(grpc_client.mClient)

	connectParam := ConnectParam{address, port,}
	status := client.Connect(connectParam)
	if status.getStatus().ErrorCode != int32(OK) {
		println("client: connect failed")
	}

	if client.IsConnected() == false {
		println("client: not connected")
		return
	}

	tableSchema := TableSchema{tableName, dimension, indexFileSize, metricType,}
	status = client.CreateTable(tableSchema)
	if status.getStatus().ErrorCode != int32(OK) {
		println("Create table failed")
	}

	hasTable := client.HasTable(tableName)
	if hasTable == false {
		println("Create table failed")
	}

	status = client.DropTable(tableName)
	hasTable = client.HasTable(tableName)
	if status.getStatus().ErrorCode != int32(OK) || hasTable == true {
		println("Drop table failed")
	}

	recordArray := make([][]float32, nb)
	var i, j int64
	for i = 0; i < nb; i++ {
		recordArray[i] = make([]float32, dimension)
		for j = 0; j < dimension; j++ {
			recordArray[i][j] = float32(i % (j+ 1))
		}
	}
	insertParam := InsertParam{tableName, "", recordArray, nil,}
	status = client.Insert(&insertParam)
	if status.getStatus().ErrorCode != int32(OK) {
		println("Insert vector failed")
	}

	queryVectors := make([][]float32, nq)
	for i = 0;i < nq;i++ {
		queryVectors[i] = make([]float32, nq)
		for j = 0; j < dimension; j++ {
			queryVectors[i][j] = float32(i % (j + 1))
		}
	}

	//var topkQueryResult TopkQueryResult
	//searchParam := SearchParam{tableName, }
	//status topkQueryResult =
}

func main() {
	address := "localhost"
	port := "19530"
	example(address, port)
}
