package tests

import (
	"context"
	"flag"
	"fmt"
	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"os"
	"strconv"
	"testing"
)

var host string
var port int64

func init()  {
	flag.StringVar(&host, "host", "10.98.0.13", "server host")
	flag.Int64Var(&port, "port", 19530, "server port")
}

func GetMilvusAddr() string {
	return host + ":" + strconv.FormatInt(port, 10)
}

func GetClient() client.Client {
	addr := GetMilvusAddr()
	client, err := client.NewGrpcClient(context.Background(), addr)
	if err != nil {
		fmt.Printf("Failed to connect %s\n", addr)
		return nil
	}
	return client
}

func teardown()  {
	fmt.Println("Start to tear down")
	ctx := context.Background()
	client := GetClient()
	collections, _ := client.ListCollections(ctx)
	for _,collection := range collections {
		client.DropCollection(ctx, collection.Name)
	}
}

func TestMain(m *testing.M)  {
	flag.Parse()
	fmt.Printf("host: %s, port:%d\n", host, port)
	code := m.Run()
	teardown()
	os.Exit(code)
}
