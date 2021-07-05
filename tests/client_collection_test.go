package tests

import (
	"context"
	"testing"

	"github.com/milvus-io/milvus-sdk-go/client"
	"github.com/milvus-io/milvus-sdk-go/entity"
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

func TestListCollections(t *testing.T) {
	c, err := client.NewGrpcClient(context.Background(), "localhost:19530")
	assert.Nil(t, err)
	assert.NotNil(t, c)
	if c != nil {
		defer c.Close()
	}
	//TODO create before list and assert exists
	_, err = c.ListCollections(context.Background())
	assert.Nil(t, err)
}

func TestCreateCollection(t *testing.T) {
	c, err := client.NewGrpcClient(context.Background(), "localhost:19530")
	assert.Nil(t, err)
	assert.NotNil(t, c)
	if c != nil {
		defer c.Close()
	}

	schema := entity.Schema{
		CollectionName: "test_go_sdk",
		AutoID:         false,
		Fields: []*entity.Field{
			{
				Name:       "int64",
				DataType:   entity.FieldTypeInt64,
				PrimaryKey: true,
			},
			{
				Name:     "vector",
				DataType: entity.FieldTypeFloatVector,
				TypeParams: map[string]string{
					"dim": "128",
				},
			},
		},
	}
	err = c.CreateCollection(context.Background(), schema, 1)
	assert.Nil(t, err)
}

func TestDescribeCollection(t *testing.T) {
	c, err := client.NewGrpcClient(context.Background(), "localhost:19530")
	assert.Nil(t, err)
	assert.NotNil(t, c)
	if c != nil {
		defer c.Close()
	}
	//TODO merge describe and create
	c.DescribeCollection(context.Background(), "test_go_sdk")
}

func TestDropCollection(t *testing.T) {
	c, err := client.NewGrpcClient(context.Background(), "localhost:19530")
	assert.Nil(t, err)
	assert.NotNil(t, c)
	if c != nil {
		defer c.Close()
	}

	c.DropCollection(context.Background(), "test_go_sdk")
}

func TestGetCollectionStatistics(t *testing.T) {
	c, err := client.NewGrpcClient(context.Background(), "localhost:19530")
	assert.Nil(t, err)
	assert.NotNil(t, c)
	if c != nil {
		defer c.Close()
	}

	c.GetCollectionStatistics(context.Background(), "test_go_sdk")
}
