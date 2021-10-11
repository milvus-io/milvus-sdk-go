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
	flag.StringVar(&host, "host", "localhost", "server host")
	flag.Int64Var(&port, "port", 19530, "server port")
}

// Generate addr to connect Milvus
func GenMilvusAddr() string {
	return host + ":" + strconv.FormatInt(port, 10)
}

// Generate an connected Milvus client
func GenClient() (client.Client, error) {
	addr := GenMilvusAddr()
	client, err := client.NewGrpcClient(context.Background(), addr)
	if err != nil {
		fmt.Printf("Failed to connect %s\n", addr)
		return nil, nil
	}
	return client, nil
}

func teardown()  {
	fmt.Println("Start to tear down")
	ctx := context.Background()
	client, _ := GenClient()
	collections, _ := client.ListCollections(ctx)
	for _,collection := range collections {
		client.DropCollection(ctx, collection.Name)
	}
	defer client.Close()
}

func TestMain(m *testing.M)  {
	flag.Parse()
	fmt.Printf("host: %s, port:%d\n", host, port)
	code := m.Run()
	teardown()
	os.Exit(code)
}
