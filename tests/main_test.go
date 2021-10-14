package tests

import (
	"context"
	"flag"
	"fmt"
	"github.com/milvus-io/milvus-sdk-go/v2/client"
	ut "github.com/milvus-io/milvus-sdk-go/v2/tests/testutil"
	"os"
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
	addr := fmt.Sprintf("%s:%d", host, port)
	return addr
}

// Generate an connected Milvus client
func GenClient(t *testing.T) client.Client {
	t.Helper()
	addr := GenMilvusAddr()
	c, err := client.NewGrpcClient(context.Background(), addr)
	if err != nil {
		t.Errorf("Failed to connect %s\n", addr)
	}
	return c
}

// Generate an collection
func GenCollection(t *testing.T, name string)  {
	t.Helper()
	c := GenClient(t)
	fields := ut.GenDefaultFields(ut.DefaultDim)
	schema := ut.GenSchema(name, false, fields)
	err := c.CreateCollection(context.Background(), schema, ut.DefaultShards)
	if err != nil {
		t.Errorf("Failed to create collection %s\n", name)
	}
}

func teardown()  {
	fmt.Println("Start to tear down")
	ctx := context.Background()
	client, _ := client.NewGrpcClient(context.Background(), GenMilvusAddr())
	collections, _ := client.ListCollections(ctx)
	for _, collection := range collections {
		client.DropCollection(ctx, collection.Name)
		fmt.Printf("Drop collection %s\n", collection.Name)
	}
	defer client.Close()
}

func TestMain(m *testing.M)  {
	flag.Parse()
	code := m.Run()
	teardown()
	os.Exit(code)
}