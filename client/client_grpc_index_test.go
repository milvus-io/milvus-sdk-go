package client

import (
	"context"
	"errors"
	"math/rand"
	"testing"
	"time"

	"github.com/golang/protobuf/proto"
	common "github.com/milvus-io/milvus-proto/go-api/commonpb"
	server "github.com/milvus-io/milvus-proto/go-api/milvuspb"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
	"github.com/stretchr/testify/assert"
)

func TestGrpcClientCreateIndex(t *testing.T) {
	ctx := context.Background()
	c := testClient(ctx, t)
	mock.SetInjection(MHasCollection, hasCollectionDefault)
	mock.SetInjection(MDescribeCollection, describeCollectionInjection(t, 0, testCollectionName, defaultSchema()))

	fieldName := `vector`
	idx, err := entity.NewIndexFlat(entity.IP)
	assert.Nil(t, err)
	if !assert.NotNil(t, idx) {
		t.FailNow()
	}
	mock.SetInjection(MCreateIndex, func(_ context.Context, raw proto.Message) (proto.Message, error) {
		req, ok := raw.(*server.CreateIndexRequest)
		if !ok {
			return BadRequestStatus()
		}
		assert.Equal(t, testCollectionName, req.GetCollectionName())
		assert.Equal(t, fieldName, req.GetFieldName())
		return SuccessStatus()
	})

	t.Run("test async create index", func(t *testing.T) {
		assert.Nil(t, c.CreateIndex(ctx, testCollectionName, fieldName, idx, true))
	})

	t.Run("test flush err", func(t *testing.T) {
		mock.SetInjection(MFlush, func(_ context.Context, raw proto.Message) (proto.Message, error) {
			resp := &server.FlushResponse{}
			s, err := BadRequestStatus()
			resp.Status = s
			return resp, err
		})
		defer mock.DelInjection(MFlush)
		assert.NotNil(t, c.CreateIndex(ctx, testCollectionName, fieldName, idx, false))
	})

	t.Run("test sync create index", func(t *testing.T) {
		buildTime := rand.Intn(900) + 100
		start := time.Now()
		flag := false
		mock.SetInjection(MGetIndexState, func(_ context.Context, raw proto.Message) (proto.Message, error) {
			req, ok := raw.(*server.GetIndexStateRequest)
			resp := &server.GetIndexStateResponse{}
			if !ok {
				s, err := BadRequestStatus()
				resp.Status = s
				return resp, err
			}
			assert.Equal(t, testCollectionName, req.CollectionName)

			resp.State = common.IndexState_InProgress
			if time.Since(start) > time.Duration(buildTime)*time.Millisecond {
				resp.State = common.IndexState_Finished
				flag = true
			}

			s, err := SuccessStatus()
			resp.Status = s
			return resp, err
		})

		assert.Nil(t, c.CreateIndex(ctx, testCollectionName, fieldName, idx, false))
		assert.True(t, flag)

		mock.DelInjection(MGetIndexState)
	})
}

func TestGrpcClientDropIndex(t *testing.T) {
	ctx := context.Background()
	c := testClient(ctx, t)
	mock.SetInjection(MHasCollection, hasCollectionDefault)
	mock.SetInjection(MDescribeCollection, describeCollectionInjection(t, 0, testCollectionName, defaultSchema()))
	assert.Nil(t, c.DropIndex(ctx, testCollectionName, "vector"))
}

func TestGrpcClientDescribeIndex(t *testing.T) {
	ctx := context.Background()
	mock.SetInjection(MHasCollection, hasCollectionDefault)
	mock.SetInjection(MDescribeCollection, describeCollectionInjection(t, 0, testCollectionName, defaultSchema()))

	c := testClient(ctx, t)

	fieldName := "vector"

	t.Run("normal describe index", func(t *testing.T) {
		mock.SetInjection(MDescribeIndex, func(_ context.Context, raw proto.Message) (proto.Message, error) {
			req, ok := raw.(*server.DescribeIndexRequest)
			resp := &server.DescribeIndexResponse{}
			if !ok {
				s, err := BadRequestStatus()
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
			s, err := SuccessStatus()
			resp.Status = s
			return resp, err
		})

		idxes, err := c.DescribeIndex(ctx, testCollectionName, fieldName)
		assert.Nil(t, err)
		assert.NotNil(t, idxes)
	})

	t.Run("Service return errors", func(t *testing.T) {
		defer mock.DelInjection(MDescribeIndex)
		mock.SetInjection(MDescribeIndex, func(_ context.Context, raw proto.Message) (proto.Message, error) {
			resp := &server.DescribeIndexResponse{}

			return resp, errors.New("mocked error")
		})

		_, err := c.DescribeIndex(ctx, testCollectionName, fieldName)
		assert.Error(t, err)

		mock.SetInjection(MDescribeIndex, func(_ context.Context, raw proto.Message) (proto.Message, error) {
			resp := &server.DescribeIndexResponse{}
			resp.Status = &common.Status{ErrorCode: common.ErrorCode_UnexpectedError}
			return resp, nil
		})

		_, err = c.DescribeIndex(ctx, testCollectionName, fieldName)
		assert.Error(t, err)
	})
}

func TestGrpcGetIndexBuildProgress(t *testing.T) {
	ctx := context.Background()
	mock.SetInjection(MHasCollection, hasCollectionDefault)
	mock.SetInjection(MDescribeCollection, describeCollectionInjection(t, 0, testCollectionName, defaultSchema()))

	tc := testClient(ctx, t)
	c := tc.(*GrpcClient) // since GetIndexBuildProgress is not exposed

	t.Run("normal get index build progress", func(t *testing.T) {
		var total, built int64

		mock.SetInjection(MGetIndexBuildProgress, func(_ context.Context, raw proto.Message) (proto.Message, error) {
			req, ok := raw.(*server.GetIndexBuildProgressRequest)
			if !ok {
				t.FailNow()
			}
			assert.Equal(t, testCollectionName, req.GetCollectionName())
			resp := &server.GetIndexBuildProgressResponse{
				TotalRows:   total,
				IndexedRows: built,
			}
			s, err := SuccessStatus()
			resp.Status = s
			return resp, err
		})

		total = rand.Int63n(1000)
		built = rand.Int63n(total)
		rt, rb, err := c.GetIndexBuildProgress(ctx, testCollectionName, testVectorField)
		assert.NoError(t, err)
		assert.Equal(t, total, rt)
		assert.Equal(t, built, rb)
	})

	t.Run("Service return errors", func(t *testing.T) {
		defer mock.DelInjection(MGetIndexBuildProgress)
		mock.SetInjection(MGetIndexBuildProgress, func(_ context.Context, raw proto.Message) (proto.Message, error) {
			_, ok := raw.(*server.GetIndexBuildProgressRequest)
			if !ok {
				t.FailNow()
			}
			resp := &server.GetIndexBuildProgressResponse{}
			return resp, errors.New("mocked error")
		})

		_, _, err := c.GetIndexBuildProgress(ctx, testCollectionName, testVectorField)
		assert.Error(t, err)

		mock.SetInjection(MGetIndexBuildProgress, func(_ context.Context, raw proto.Message) (proto.Message, error) {
			_, ok := raw.(*server.GetIndexBuildProgressRequest)
			if !ok {
				t.FailNow()
			}
			resp := &server.GetIndexBuildProgressResponse{}
			resp.Status = &common.Status{
				ErrorCode: common.ErrorCode_UnexpectedError,
			}
			return resp, nil
		})
		_, _, err = c.GetIndexBuildProgress(ctx, testCollectionName, testVectorField)
		assert.Error(t, err)
	})

}
