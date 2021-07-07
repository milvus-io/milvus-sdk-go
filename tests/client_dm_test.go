package tests

import (
	"context"
	"testing"
	"time"

	"github.com/milvus-io/milvus-sdk-go/client"
	"github.com/milvus-io/milvus-sdk-go/entity"
	"github.com/stretchr/testify/assert"
)

func TestSearch(t *testing.T) {
	c := getDefaultClient(t)
	if c != nil {
		defer c.Close()
	}

	vectors := generateTestCollection(t, c)

	waitRowCountChanged(t, c)
	c.LoadCollection(context.Background(), testCollectionName, false)
	results, err := c.Search(context.Background(), testCollectionName, []string{}, "int64 > 0", []string{"int64"}, []entity.Vector{entity.FloatVector(vectors[0])},
		testVectorField, entity.L2, 10, map[string]string{
			"nprobe": "10",
		})

	assert.Nil(t, err)
	for _, result := range results {
		t.Logf("result count: %d, ids %v fields %v", result.ResultCount, result.IDs, result.Fields)
	}
}

func waitRowCountChanged(t *testing.T, c client.Client) {
	start := time.Now()
	f := func() bool {
		ctx := context.Background()
		m, err := c.GetCollectionStatistics(ctx, testCollectionName)
		assert.Nil(t, err)
		for k, v := range m {
			t.Log(k, v)
		}
		return m["row_count"] != "0"
	}
	for !f() {
		time.Sleep(time.Millisecond * 100)
	}
	t.Log(time.Since(start))

}
