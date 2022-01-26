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

var (
	defaultMilvusAddr = `localhost:19530`

	testCollectionName = `test_sdk_go` // collection name used for testing
	testPrimaryField   = `int64`       // default primary key name
	testVectorField    = `vector`      // default vector field name
	testVectorDim      = 128
	testShardsNum      = 1
)

func getDefaultClient(t *testing.T) client.Client {
	c, err := client.NewGrpcClient(context.Background(), defaultMilvusAddr)
	assert.Nil(t, err)
	assert.NotNil(t, c)
	if c == nil {
		t.FailNow()
	}
	return c
}

func generateTestCollection(t *testing.T, c client.Client) [][]float32 {
	if c == nil {
		t.FailNow()
		return nil
	}

	ctx := context.Background()
	has, err := c.HasCollection(ctx, testCollectionName)
	assert.Nil(t, err)
	if has { // maybe last test crashed, do clean up
		assert.Nil(t, c.DropCollection(ctx, testCollectionName))
	}

	assert.Nil(t, c.CreateCollection(ctx, defaultSchema(), int32(testShardsNum)))

	vector := generateFloatVector(4096, testVectorDim)
	_, err = c.Insert(ctx, testCollectionName, "", // use default partition
		entity.NewColumnFloatVector(testVectorField, testVectorDim, vector))
	if err != nil {
		t.Log(err.Error())
	}
	c.Flush(ctx, testCollectionName, false)
	return vector
}

func defaultSchema() *entity.Schema {
	return &entity.Schema{
		CollectionName: testCollectionName,
		AutoID:         false,
		Fields: []*entity.Field{
			{
				Name:       testPrimaryField,
				DataType:   entity.FieldTypeInt64,
				PrimaryKey: true,
				AutoID:     true,
			},
			{
				Name:     testVectorField,
				DataType: entity.FieldTypeFloatVector,
				TypeParams: map[string]string{
					entity.TYPE_PARAM_DIM: fmt.Sprintf("%d", testVectorDim),
				},
				IndexParams: map[string]string{
					"metric_type": "L2",
				},
			},
		},
	}
}

func generateFloatVector(num, dim int) [][]float32 {
	rand.Seed(time.Now().Unix())
	r := make([][]float32, 0, num)
	for i := 0; i < num; i++ {
		v := make([]float32, 0, dim)
		for j := 0; j < dim; j++ {
			v = append(v, rand.Float32())
		}
		r = append(r, v)
	}
	return r
}
