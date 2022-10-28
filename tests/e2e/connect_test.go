package e2e

import (
	"context"
	"testing"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/stretchr/testify/assert"
)

func TestConnect(t *testing.T) {
	c, err := client.NewGrpcClient(context.Background(), "localhost:19530")
	assert.Nil(t, err)
	assert.NotNil(t, c)
	if c != nil {
		defer c.Close()
	}
}
