package tests

import (
	"context"
	"flag"
	"fmt"
	"os"
	"testing"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
)

var host string
var port int64

func init() {
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

func teardown() {
	fmt.Println("Start to tear down")
	ctx := context.Background()
	c, _ := client.NewGrpcClient(context.Background(), GenMilvusAddr())
	defer c.Close()
	for _, cname := range testCollections {
		c.DropCollection(ctx, cname)
		fmt.Printf("Drop collection %s\n", cname)
	}
}

func TestMain(m *testing.M) {
	flag.Parse()
	code := m.Run()
	teardown()
	os.Exit(code)
}
