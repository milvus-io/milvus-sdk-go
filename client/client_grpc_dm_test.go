package client

import (
	"context"
	"math/rand"
	"testing"

	"github.com/golang/protobuf/proto"
	"github.com/milvus-io/milvus-sdk-go/entity"
	"github.com/milvus-io/milvus-sdk-go/internal/proto/schema"
	"github.com/milvus-io/milvus-sdk-go/internal/proto/server"
	"github.com/stretchr/testify/assert"
)

func TestGrpcClientInsert(t *testing.T) {
	ctx := context.Background()

	c := testClient(ctx, t)

	vector := generateFloatVector(4096, testVectorDim)
	mock.setInjection(mInsert, func(_ context.Context, raw proto.Message) (proto.Message, error) {
		req, ok := raw.(*server.InsertRequest)
		resp := &server.MutationResult{}
		if !ok {
			s, err := badRequestStatus()
			resp.Status = s
			return resp, err
		}
		assert.EqualValues(t, 4096, req.GetNumRows())
		assert.Equal(t, testCollectionName, req.GetCollectionName())
		intIds := &schema.IDs_IntId{
			IntId: &schema.LongArray{
				Data: make([]int64, 4096),
			},
		}
		resp.IDs = &schema.IDs{
			IdField: intIds,
		}
		s, err := successStatus()
		resp.Status = s
		return resp, err
	})
	_, err := c.Insert(ctx, testCollectionName, "", // use default partition
		entity.NewColumnFloatVector(testVectorField, testVectorDim, vector))

	assert.Nil(t, err)
}

func TestGrpcClientFlush(t *testing.T) {

	ctx := context.Background()

	c := testClient(ctx, t)

	assert.Nil(t, c.Flush(ctx, testCollectionName, false))
	assert.Nil(t, c.Flush(ctx, testCollectionName, true))
}

func TestGrpcSearch(t *testing.T) {

	ctx := context.Background()

	c := testClient(ctx, t)

	vectors := generateFloatVector(4096, testVectorDim)

	results, err := c.Search(context.Background(), testCollectionName, []string{}, "int64 > 0", []string{"int64"}, []entity.Vector{entity.FloatVector(vectors[0])},
		testVectorField, entity.L2, 10, map[string]string{
			"nprobe": "10",
		})

	assert.Nil(t, err)
	assert.NotNil(t, results)
}

func generateFloatVector(num, dim int) [][]float32 {
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
