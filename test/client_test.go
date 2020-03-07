package test

import (
	"github.com/milvus-io/milvus-sdk-go/milvus"
	"testing"
)

var TABLENAME string = "go_test"
var client milvus.MilvusClient = GetClient()

func GetClient() milvus.MilvusClient {
	var grpcClient milvus.Milvusclient
	client := milvus.NewMilvusClient(grpcClient.Instance)
	connectParam := milvus.ConnectParam{"127.0.0.1", "19530"}
	err := client.Connect(connectParam)
	if err != nil {
		println("Connect failed")
		return nil
	}
	return client
}

func CreateTable() error {
	boolReply, status, err := client.HasTable(TABLENAME)
	if boolReply == true {
		return err
	}

	tableSchema := milvus.TableSchema{TABLENAME, 128, 1024, int64(milvus.L2), nil}
	status, err = client.CreateTable(tableSchema)
	if err != nil {
		return err
	}
	if !status.Ok() {
		println("CreateTable failed")
	}
	return err
}

func TestConnection(t *testing.T) {
	var grpcClient milvus.Milvusclient
	testClient := milvus.NewMilvusClient(grpcClient.Instance)
	connectParam := milvus.ConnectParam{"127.0.0.1", "19530"}
	err := testClient.Connect(connectParam)
	if err != nil {
		t.Error("Connect error: " + err.Error())
	}

	// test wrong uri connect
	connectParam = milvus.ConnectParam{"12345", "111"}
	err = testClient.Connect(connectParam)
	if err == nil {
		t.Error("Use wrong uri to connect, return true")
	}
}

func TestTable(t *testing.T) {
	param := milvus.TableSchema{"test_1", 128, 1024, int64(milvus.L2), nil}
	status, err := client.CreateTable(param)
	if err != nil {
		t.Error("CreateTable error")
	}

	if !status.Ok() {
		t.Error("CreateTable return status wrong!")
	}

	// test ShowTables
	tables, status, err := client.ShowTables()
	if err != nil {
		t.Error("ShowTables error")
		return
	}
	if !status.Ok() {
		t.Error("ShowTables status check error")
	}
	if len(tables) != 1 && tables[0] != TABLENAME {
		t.Error("ShowTables result check error")
	}

	// test normal hastable
	hasTable, status, err := client.HasTable("test_1")
	if err != nil {
		t.Error("HasTable error")
		return
	}

	if !status.Ok() {
		t.Error("HasTable status check error")
	}

	if !hasTable {
		t.Error("HasTable result check error")
	}

	// test HasTable with table not exist
	hasTable, status, err = client.HasTable("aaa")
	if err != nil {
		t.Error("HasTable error")
	}

	//if !status.Ok() {
	//	t.Error("HasTable status check error: " + status.GetMessage())
	//}

	if hasTable == true {
		t.Error("HasTable result check error")
	}

	// test DropTable
	status, err = client.DropTable("test_1")
	if err != nil {
		t.Error("DropTable error")
	}

	if !status.Ok() {
		t.Error("DropTable status check error")
	}

	hasTable, status, err = client.HasTable("test_1")
	if hasTable == true {
		t.Error("DropTable result check error")
	}

	// test DropTable with table not exist
	status, err = client.DropTable("aaa")
	if err != nil {
		t.Error("DropTable error")
	}

	if status.Ok() {
		t.Error("DropTable status check error")
	}
}

func TestVector(t *testing.T) {
	err := CreateTable()
	if err != nil {
		t.Error("Create table error")
	}
	// test insert
	var i, j int

	nb := 10000
	dimension := 128
	records := make([]milvus.RowRecord, nb)
	recordArray := make([][]float32, 10000)
	for i = 0; i < nb; i++ {
		recordArray[i] = make([]float32, dimension)
		for j = 0; j < dimension; j++ {
			recordArray[i][j] = float32(i % (j + 1))
		}
		records[i].FloatData = recordArray[i]
	}
	insertParam := milvus.InsertParam{TABLENAME, "", records, nil, nil}
	status, err := client.Insert(&insertParam)
	if err != nil {
		t.Error("Insert error")
		return
	}

	if !status.Ok() {
		t.Error("Insert status check error")
	}

	// test ShowTableInfos
	tableInfo, status, err := client.ShowTableInfo(TABLENAME)
	if err != nil {
		t.Error("ShowTableInfo error")
		return
	}

	if !status.Ok() {
		t.Error("ShowTableInfo status check error")
	}

	if tableInfo.TotalRowCount == 0 {
		//t.Error("ShowTableInfo result check error")
	}

	// test GetVectorIds
	getVectorIDsParam := milvus.GetVectorIDsParam{TABLENAME, ""}
	vectorIDs, status, err := client.GetVectorIDs(getVectorIDsParam)
	if err != nil {
		t.Error("GetVectorIDs error")
		return
	}

	if len(vectorIDs) == 0 {
		//t.Error("GetVectorIDs result check error")
		//return
	}

	// test GetVectorById
	rowRecord, status, err := client.GetVectorByID(TABLENAME, 0)
	if err != nil {
		t.Error("GetVectorByID error")
		return
	}
	//if !status.Ok() {
	//	t.Error("GetVectorByID status check error")
	//}
	if len(rowRecord.FloatData) != 128 {
		//t.Error("GetVectorByID result check error")
	}

	// test DeleteByID
}

func TestIndex(t *testing.T) {
	extraParams := make([]milvus.KeyValuePair, 1)
	extraParams[0].Key = "params"
	extraParams[0].Value = "{\"nlist\" : 16384}"
	indexParam := milvus.IndexParam{TABLENAME, milvus.IVFFLAT, extraParams}
	status, err := client.CreateIndex(&indexParam)
	if err != nil {
		t.Error("CreateIndex error")
	}
	if !status.Ok() {
		t.Error("CreateIndex status check error")
	}

	// test DescribeIndex
	indexInfo, status, err := client.DescribeIndex(TABLENAME)
	if err != nil {
		t.Error("DescribeIndex error")
		return
	}
	if !status.Ok() {
		t.Error("DescribeIndex status check error")
	}
	if indexInfo.TableName != TABLENAME || indexInfo.IndexType != milvus.IVFFLAT {
		t.Error("DescribeIndex result chck error")
	}

	// test DropIndex
	status, err = client.DropIndex(TABLENAME)
	if err != nil {
		t.Error("DropIndex error")
		return
	}

	if !status.Ok() {
		t.Error("DropIndex status check erro")
	}

	status, err = client.CreateIndex(&indexParam)
	if err != nil {
		t.Error("CreateIndex error")
	}
	if !status.Ok() {
		t.Error("CreateIndex status check error")
	}
}

func TestSearch(t *testing.T) {
	var i, j int
	//Construct query vectors
	nq := 10
	dimension := 128
	topk := 10
	queryRecords := make([]milvus.RowRecord, nq)
	queryVectors := make([][]float32, nq)
	for i = 0; i < nq; i++ {
		queryVectors[i] = make([]float32, dimension)
		for j = 0; j < dimension; j++ {
			queryVectors[i][j] = float32(i % (j + 1))
		}
		queryRecords[i].FloatData = queryVectors[i]
	}

	var topkQueryResult milvus.TopkQueryResult
	kvPair := make([]milvus.KeyValuePair, 1)
	kvPair[0].Key = "params"
	kvPair[0].Value = "{\"nprobe\" : 32}"
	searchParam := milvus.SearchParam{TABLENAME, queryRecords, int64(topk), nil, kvPair}
	topkQueryResult, status, err := client.Search(searchParam)
	if err != nil {
		t.Error(err.Error())
	}
	if !status.Ok() {
		t.Error("Search status check error")
	}
	if len(topkQueryResult.QueryResultList) != nq {
		t.Error("Search result check error")
	}
}

func TestCmd(t *testing.T) {
	// test ServerStatus
	serverStatus, status, err := client.ServerStatus()
	if err != nil {
		t.Error("ServerStatus error")
		return
	}
	if !status.Ok() {
		t.Error("ServerStatus status check error")
	}
	if serverStatus != "server alive" {
		t.Error("ServerStatus result check error: " + serverStatus)
	}

	// test ServerVersion
	serverVersion, status, err := client.ServerVersion()
	if err != nil {
		t.Error("ServerVersion error")
		return
	}
	if !status.Ok() {
		t.Error("ServerVersion status check error")
	}
	if len(serverVersion) == 0 {
		t.Error("ServerVersion result check error")
	}

	// test SetConfig and GetConfig
	nodeName := "cache_config.cpu_cache_capacity"
	nodeValue := "2"
	status, err = client.SetConfig(nodeName, nodeValue)
	if err != nil {
		t.Error("SetConfig error")
	}
	if !status.Ok() {
		t.Error("SetConfig status check error: " + status.GetMessage())
	}

	value, status, err := client.GetConfig(nodeName)
	if err != nil {
		t.Error("GetConfig error")
		return
	}
	if !status.Ok() {
		t.Error("GetConfig status check error")
	}
	if value != nodeValue {
		t.Error("GetConfig or SetConfig result check error")
	}
}

func TestPartition(t *testing.T) {
	// test CreatePartition
	partitionTag := "part_1"
	status, err := client.CreatePartition(milvus.PartitionParam{TABLENAME, partitionTag})
	if err != nil {
		t.Error("CreatePartition error")
		return
	}
	if !status.Ok() {
		t.Error("CreatePartition status check error")
	}

	// test ShowPartitions
	partitionParam, status, err := client.ShowPartitions(TABLENAME)
	if !status.Ok() {
		t.Error("ShowPartitions status check error")
	}
	if len(partitionParam) != 1 && partitionParam[0].PartitionTag != partitionTag {
		//t.Error("ShowPartitions result check error")
	}

	// test DropPartition
	status, err = client.DropPartition(milvus.PartitionParam{TABLENAME, partitionTag})
	if !status.Ok() {
		t.Error("DropPartition status check error")
	}
}

func TestFlush(t *testing.T) {
	tableNameArray := make([]string, 1)
	tableNameArray[0] = TABLENAME
	status, err := client.Flush(tableNameArray)
	if err != nil {
		t.Error("Flush error")
	}
	if !status.Ok() {
		t.Error("Flush status check error")
	}
}

func TestCompact(t *testing.T) {
	status, err := client.Compact(TABLENAME)
	if err != nil {
		t.Error("Compact error")
		return
	}
	if !status.Ok() {
		t.Error("Compact status check error")
	}
}
