package tests

import (
	"context"
	"testing"

	"github.com/milvus-io/milvus-sdk-go/client"
	"github.com/stretchr/testify/assert"
)

func TestShowPartitions(t *testing.T) {
	c, err := client.NewGrpcClient(context.Background(), "localhost:19530")
	assert.Nil(t, err)
	assert.NotNil(t, c)
	if c != nil {
		defer c.Close()
	}

	c.ShowPartitions(context.Background(), "list_collections_ggIQr3xO")
}

func TestCreatePartition(t *testing.T) {
	c, err := client.NewGrpcClient(context.Background(), "localhost:19530")
	assert.Nil(t, err)
	assert.NotNil(t, c)
	if c != nil {
		defer c.Close()
	}

	c.CreatePartition(context.Background(), "list_collections_ggIQr3xO", "_dim_part")
}

func TestDropPartition(t *testing.T) {
	c, err := client.NewGrpcClient(context.Background(), "localhost:19530")
	assert.Nil(t, err)
	assert.NotNil(t, c)
	if c != nil {
		defer c.Close()
	}

	c.DropPartition(context.Background(), "list_collections_ggIQr3xO", "_dim_part")
}
