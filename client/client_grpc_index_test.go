package client

import (
	"context"
	"math/rand"
	"testing"
	"time"

	"github.com/golang/protobuf/proto"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
	"github.com/milvus-io/milvus-sdk-go/v2/internal/proto/common"
	"github.com/milvus-io/milvus-sdk-go/v2/internal/proto/server"
	"github.com/stretchr/testify/assert"
)

func TestGrpcClientCreateIndex(t *testing.T) {
	ctx := context.Background()
	c := testClient(ctx, t)

	fieldName := `vector`
	idx, err := entity.NewIndexFlat(entity.IP, 1024)
	assert.Nil(t, err)
	if !assert.NotNil(t, idx) {
		t.FailNow()
	}
	mock.setInjection(mCreateIndex, func(_ context.Context, raw proto.Message) (proto.Message, error) {
		req, ok := raw.(*server.CreateIndexRequest)
		if !ok {
			return badRequestStatus()
		}
		assert.Equal(t, testCollectionName, req.GetCollectionName())
		assert.Equal(t, fieldName, req.GetFieldName())
		return successStatus()
	})

	t.Run("test async create index", func(t *testing.T) {
		assert.Nil(t, c.CreateIndex(ctx, testCollectionName, fieldName, idx, true))
	})

	t.Run("test sync create index", func(t *testing.T) {
		buildTime := rand.Intn(900) + 100
		start := time.Now()
		flag := false
		mock.setInjection(mGetIndexState, func(_ context.Context, raw proto.Message) (proto.Message, error) {
			req, ok := raw.(*server.GetIndexStateRequest)
			resp := &server.GetIndexStateResponse{}
			if !ok {
				s, err := badRequestStatus()
				resp.Status = s
				return resp, err
			}
			assert.Equal(t, testCollectionName, req.CollectionName)

			resp.State = common.IndexState_InProgress
			if time.Since(start) > time.Duration(buildTime)*time.Millisecond {
				resp.State = common.IndexState_Finished
				flag = true
			}

			s, err := successStatus()
			resp.Status = s
			return resp, err
		})

		assert.Nil(t, c.CreateIndex(ctx, testCollectionName, fieldName, idx, false))
		assert.True(t, flag)

		mock.delInjection(mGetIndexState)
	})
}

func TestGrpcClientDropIndex(t *testing.T) {
	ctx := context.Background()
	c := testClient(ctx, t)

	assert.Nil(t, c.DropIndex(ctx, testCollectionName, "vector"))
}

func TestGrpcClientDescribeIndex(t *testing.T) {
	ctx := context.Background()

	c := testClient(ctx, t)

	fieldName := "vector"

	mock.setInjection(mDescribeIndex, func(_ context.Context, raw proto.Message) (proto.Message, error) {
		req, ok := raw.(*server.DescribeIndexRequest)
		resp := &server.DescribeIndexResponse{}
		if !ok {
			s, err := badRequestStatus()
			resp.Status = s
			return resp, err
		}
		assert.Equal(t, fieldName, req.GetFieldName())
		assert.Equal(t, testCollectionName, req.GetCollectionName())
		resp.IndexDescriptions = []*server.IndexDescription{
			{
				IndexName: "_default",
				IndexID:   1,
				FieldName: req.GetFieldName(),
				Params: entity.MapKvPairs(map[string]string{
					"nlist":       "1024",
					"metric_type": "IP",
				}),
			},
		}
		s, err := successStatus()
		resp.Status = s
		return resp, err
	})

	idxes, err := c.DescribeIndex(ctx, testCollectionName, fieldName)
	assert.Nil(t, err)
	assert.NotNil(t, idxes)
}
