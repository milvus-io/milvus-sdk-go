package tests

import (
	"context"
	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestClientConnect(t *testing.T)  {
	client, err := client.NewGrpcClient(context.Background(), GetMilvusAddr())
	assert.Nil(t, err)
	assert.NotNil(t, client)
	t.Log("client connected")
}

func TestConnectRepeat(t *testing.T)  {
	c, err := client.NewGrpcClient(context.Background(), GetMilvusAddr())
	assert.Nil(t, err)
	assert.NotNil(t, c)
	c1, err1 := client.NewGrpcClient(context.Background(), GetMilvusAddr())
	assert.Nil(t, err1)
	assert.NotNil(t, c1)

}

