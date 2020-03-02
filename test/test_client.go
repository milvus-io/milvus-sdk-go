package test

import (
	"github.com/milvus-io/milvus-sdk-go/milvus"
	"testing"
)

var TABLENAME string = "go_test"

func GetClient() milvus.MilvusClient {
	var grpcClient milvus.Milvusclient
	client := milvus.NewMilvusClient(grpcClient.Instance)
	connectParam := milvus.ConnectParam{"127.0.0.1", "19530"}
	err := client.Connect(connectParam)
	if err != nil {
		println("Connect failed")
	}
}

func CreateTable() error {
	client := GetClient()
	tableSchema := milvus.TableSchema{TABLENAME, 128, 1024, int64(milvus.L2), nil}
	status, err := client.CreateTable()
	if err != nil {
		return err
	}
	if !status.Ok() {
		println("CreateTable failed")
	}
}

func TestTrueConnection(t *testing.T) {
	var grpcClient milvus.Milvusclient
	client := milvus.NewMilvusClient(grpcClient.Instance)
	connectParam := milvus.ConnectParam{"127.0.0.1", "19530"}
	err := client.Connect(connectParam)
	if err != nil {
		t.Error("Connect error: " + err.Error())
	}

	// test wrong uri connect
	connectParam = milvus.ConnectParam{"12345", "111"}
	err = client.Connect(connectParam)
	if err == nil {
		t.Error("Use wrong uri to connect, return true")
	}
}

func TestCreateTable(t *testing.T) {
	client := GetClient()
	param := milvus.TableSchema{"test_1", 128, 1024, int64(milvus.L2), nil}
	status, err := client.CreateTable(param)
	if err != nil {
		t.Error("CreateTable error")
	}

	if !status.Ok() {
		t.Error("CreateTable return status wrong!")
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
		t.Error("Table does not exist")
	}

	// test HasTable with table not exist
	hasTable, status, err = client.HasTable("aaa")
	if err != nil {
		t.Error("HasTable error")
	}

	if !status.Ok() {
		t.Error("HasTable status check error")
	}

	if hasTable == true {
		t.Error("HasTable check error")
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
		t.Error("Check DropTable error")
	}

	// test DropTable with table not exist
	status, err = client.DropTable("aaa")
	if err != nil {
		t.Error("DropTable error")
	}

	if !status.Ok() {
		t.Error("DropTable status check error")
	}
}

func TestVector(t *testing.T) {
	client := GetClient()
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
	}

	if !status.Ok() {
		t.Error("Insert status check error")
	}

	// test DeleteByID

}

func TestIndex(t *testing.T) {

}

func TestSearch(t *testing.T) {

}
