package tests

import (
	"context"
	"fmt"
	"math/rand"
	"testing"
	"time"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
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
	// TODO create before list and assert exists
	_, err = c.ListCollections(context.Background())
	assert.Nil(t, err)
}

func TestCreateCollection(t *testing.T) {
	rand.Seed(time.Now().UnixNano())
	c, err := client.NewGrpcClient(context.Background(), "localhost:19530")
	assert.Nil(t, err)
	assert.NotNil(t, c)
	if c != nil {
		defer c.Close()
	}

	cname := generateCollectionName()
	schema := &entity.Schema{
		CollectionName: cname,
		Fields: []*entity.Field{
			{
				Name:       fmt.Sprintf("int64_%v", rand.Intn(10)),
				DataType:   entity.FieldTypeInt64,
				PrimaryKey: true,
			},
			{
				Name:     "vector",
				DataType: entity.FieldTypeFloatVector,
				TypeParams: map[string]string{
					entity.TypeParamDim: "128",
				},
			},
		},
	}
	shards := rand.Int31n(3)
	err = c.CreateCollection(context.Background(), schema, shards)
	assert.Nil(t, err)
}

func TestDescribeCollection(t *testing.T) {
	c, err := client.NewGrpcClient(context.Background(), "localhost:19530")
	assert.Nil(t, err)
	assert.NotNil(t, c)
	if c != nil {
		defer c.Close()
	}
	cname := generateCollectionName()
	schema := generateSchema()
	generateCollection(t, c, cname, schema, false)

	// TODO merge describe and create
	coll, err := c.DescribeCollection(context.Background(), cname)
	if assert.Nil(t, err) {
		t.Logf("schema -- name: %s\n", coll.Name)
		for _, field := range coll.Schema.Fields {
			t.Logf("schema -- field: %s data type: %v, is primary: %v\n", field.Name, field.DataType, field.PrimaryKey)
		}
	}
}

func TestDropCollection(t *testing.T) {
	c, err := client.NewGrpcClient(context.Background(), "localhost:19530")
	assert.Nil(t, err)
	assert.NotNil(t, c)
	if c != nil {
		defer c.Close()
	}

	cname := generateCollectionName()
	schema := generateSchema()
	generateCollection(t, c, cname, schema, false)

	c.DropCollection(context.Background(), cname)
}

func TestGetCollectionStatistics(t *testing.T) {
	c, err := client.NewGrpcClient(context.Background(), "localhost:19530")
	assert.Nil(t, err)
	assert.NotNil(t, c)
	if c != nil {
		defer c.Close()
	}

	cname := generateCollectionName()
	schema := generateSchema()
	generateCollection(t, c, cname, schema, false)

	m, err := c.GetCollectionStatistics(context.Background(), cname)
	if assert.Nil(t, err) {
		for k, v := range m {
			t.Log(k, v)
		}
	}
}
