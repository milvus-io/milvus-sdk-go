package tests

import (
	"context"
	"fmt"
	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/stretchr/testify/assert"
	"reflect"
	"testing"
)

func TestClientConnect(t *testing.T)  {
	c, err := client.NewGrpcClient(context.Background(), GenMilvusAddr())
	assert.NoError(t, err)
	tp := reflect.TypeOf(c)
	t.Logf("Client type is: %s\n", tp)
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
	addrTests := []struct{
		addr string
	}{
		{fmt.Sprintf("%s%d", host, port)},
		{host},
		{fmt.Sprintf("%s:-1", host)},
		{fmt.Sprintf("%s:%d_", host, port)},
		{fmt.Sprintf(" :%d_", port)},
		{fmt.Sprintf("中文:%d_", port)},
	}
	errString := "context deadline exceeded"
	for _, test := range addrTests {
		t.Log(test.addr)
		_, err := client.NewGrpcClient(context.Background(), test.addr)
		assert.EqualError(t, err, errString)
	}
}

func TestClose(t *testing.T) {
	c := GenClient(t)
	res := c.Close()
	assert.NoError(t, res)
	_, err1 := c.ListCollections(context.Background())
	errString := "rpc error: code = Canceled desc = grpc: the client connection is closing"
	assert.EqualError(t, err1, errString)
}

func TestCloseRepeatedly(t *testing.T)  {
	c := GenClient(t)
	res := c.Close()
	assert.NoError(t, res)
	res1 := c.Close()
	assert.NoError(t, res1)
}