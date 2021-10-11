package tests

import (
	"context"
	"fmt"
	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/stretchr/testify/assert"
	"reflect"
	"strconv"
	"testing"
)

func TestClientConnect(t *testing.T)  {
	c, err := client.NewGrpcClient(context.Background(), GenMilvusAddr())
	assert.NoError(t, err)
	tp := reflect.TypeOf(c)
	fmt.Printf("Client type is: %s\n", tp)
	//assert.IsType(t, *client.grpcClient, c)
	defer c.Close()
}

func TestConnectRepeatedly(t *testing.T)  {
	c, err := client.NewGrpcClient(context.Background(), GenMilvusAddr())
	assert.NoError(t, err)
	c1, err1 := client.NewGrpcClient(context.Background(), GenMilvusAddr())
	assert.NoError(t, err1)
	assert.ObjectsAreEqualValues(c, c1)
	defer c.Close()
}

func TestConnectInvalid(t *testing.T)  {
	addrs := []struct{
		addrString string
	}{
		{host + strconv.FormatInt(port, 10)},
		{host},
		{host + ":" + "-1"},
		{host + ":" + strconv.FormatInt(port, 10) + "_"},
		{"" + ":" + strconv.FormatInt(port, 10)},
		{"中文" + ":" + strconv.FormatInt(port, 10)},
	}
	errString := "context deadline exceeded"
	for _, addr := range addrs {
		t.Log(addr.addrString)
		_, err := client.NewGrpcClient(context.Background(), addr.addrString)
		assert.EqualError(t, err, errString)
	}
}

func TestClose(t *testing.T) {
	c, _ := GenClient()
	res := c.Close()
	assert.NoError(t, res)
	_, err1 := c.ListCollections(context.Background())
	errString := "rpc error: code = Canceled desc = grpc: the client connection is closing"
	assert.EqualError(t, err1, errString)
}

func TestCloseRepeatedly(t *testing.T)  {
	c, _ := GenClient()
	res := c.Close()
	assert.NoError(t, res)
	res1 := c.Close()
	assert.NoError(t, res1)
}

