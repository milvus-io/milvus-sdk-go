package test

import (
	"github.com/milvus-io/milvus-sdk-go/milvus"
	"testing"
)

func TestTrueConnection(t *testing.T) {
	var grpcClient milvus.Milvusclient
	client := milvus.NewMilvusClient(grpcClient.Instance)
	connectParam := milvus.ConnectParam{"127.0.0.1", "19530"}
	status, err := client.Connect(connectParam)
}

func TestTable(t *testing.T) {

}

func TestVector(t *testing.T) {

}

func TestSearch(t *testing.T) {

}
