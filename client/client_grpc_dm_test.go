package client

import (
	"context"
	"math/rand"
	"testing"
	"time"

	"github.com/golang/protobuf/proto"
	"github.com/milvus-io/milvus-sdk-go/entity"
	"github.com/milvus-io/milvus-sdk-go/internal/proto/common"
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

	t.Run("test async flush", func(t *testing.T) {
		assert.Nil(t, c.Flush(ctx, testCollectionName, true))
	})

	t.Run("test sync flush", func(t *testing.T) {
		// 1~10 segments
		segCount := rand.Intn(10) + 1
		segments := make([]int64, 0, segCount)
		for i := 0; i < segCount; i++ {
			segments = append(segments, rand.Int63())
		}
		// 500ms ~ 2s
		flushTime := 500 + rand.Intn(1500)
		start := time.Now()
		flag := false
		mock.setInjection(mFlush, func(_ context.Context, raw proto.Message) (proto.Message, error) {
			req, ok := raw.(*server.FlushRequest)
			resp := &server.FlushResponse{}
			if !ok {
				s, err := badRequestStatus()
				resp.Status = s
				return resp, err
			}
			assert.ElementsMatch(t, []string{testCollectionName}, req.GetCollectionNames())

			resp.CollSegIDs = make(map[string]*schema.LongArray)
			resp.CollSegIDs[testCollectionName] = &schema.LongArray{
				Data: segments,
			}

			s, err := successStatus()
			resp.Status = s
			return resp, err
		})
		mock.setInjection(mGetPersistentSegmentInfo, func(_ context.Context, raw proto.Message) (proto.Message, error) {
			req, ok := raw.(*server.GetPersistentSegmentInfoRequest)
			resp := &server.GetPersistentSegmentInfoResponse{}
			if !ok {
				s, err := badRequestStatus()
				resp.Status = s
				return resp, err
			}
			assert.Equal(t, testCollectionName, req.GetCollectionName())

			state := common.SegmentState_Flushing
			if time.Since(start) > time.Duration(flushTime)*time.Millisecond {
				state = common.SegmentState_Flushed
				flag = true
			}

			for _, segID := range segments {
				resp.Infos = append(resp.Infos, &server.PersistentSegmentInfo{
					SegmentID: segID,
					State:     state,
				})
			}

			s, err := successStatus()
			resp.Status = s
			return resp, err
		})
		assert.Nil(t, c.Flush(ctx, testCollectionName, false))
		assert.True(t, flag)
	})
}

func TestGrpcSearch(t *testing.T) {

	ctx := context.Background()

	c := testClient(ctx, t)

	vectors := generateFloatVector(4096, testVectorDim)
	sp, err := entity.NewIndexFlatSearchParam(10)
	assert.Nil(t, err)
	assert.NotNil(t, sp)
	results, err := c.Search(ctx, testCollectionName, []string{}, "int64 > 0", []string{"int64"}, []entity.Vector{entity.FloatVector(vectors[0])},
		testVectorField, entity.L2, 10, sp)

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
