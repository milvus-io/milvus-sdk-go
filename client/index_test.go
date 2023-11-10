package client

import (
	"context"
	"math/rand"
	"testing"
	"time"

	"github.com/cockroachdb/errors"

	"github.com/golang/protobuf/proto"
	"github.com/milvus-io/milvus-proto/go-api/v2/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v2/milvuspb"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
	"github.com/stretchr/testify/assert"
)

func TestGrpcClientCreateIndex(t *testing.T) {
	ctx := context.Background()
	c := testClient(ctx, t)
	mockServer.SetInjection(MHasCollection, hasCollectionDefault)
	mockServer.SetInjection(MDescribeCollection, describeCollectionInjection(t, 0, testCollectionName, defaultSchema()))

	fieldName := `vector`
	idx, err := entity.NewIndexFlat(entity.IP)
	assert.Nil(t, err)
	if !assert.NotNil(t, idx) {
		t.FailNow()
	}
	mockServer.SetInjection(MCreateIndex, func(_ context.Context, raw proto.Message) (proto.Message, error) {
		req, ok := raw.(*milvuspb.CreateIndexRequest)
		if !ok {
			return BadRequestStatus()
		}
		assert.Equal(t, testCollectionName, req.GetCollectionName())
		assert.Equal(t, fieldName, req.GetFieldName())
		return SuccessStatus()
	})

	t.Run("test async create index", func(t *testing.T) {
		assert.Nil(t, c.CreateIndex(ctx, testCollectionName, fieldName, idx, true, WithIndexMsgBase(&commonpb.MsgBase{})))
	})

	t.Run("test sync create index", func(t *testing.T) {
		buildTime := rand.Intn(900) + 100
		start := time.Now()
		flag := false
		mockServer.SetInjection(MDescribeIndex, func(_ context.Context, raw proto.Message) (proto.Message, error) {
			req, ok := raw.(*milvuspb.DescribeIndexRequest)
			resp := &milvuspb.DescribeIndexResponse{}
			if !ok {
				s, err := BadRequestStatus()
				resp.Status = s
				return resp, err
			}
			assert.Equal(t, testCollectionName, req.CollectionName)
			assert.Equal(t, "test-index", req.IndexName)

			resp.IndexDescriptions = []*milvuspb.IndexDescription{
				{
					IndexName: req.GetIndexName(),
					FieldName: req.GetIndexName(),
					State:     commonpb.IndexState_InProgress,
				},
			}
			if time.Since(start) > time.Duration(buildTime)*time.Millisecond {
				resp.IndexDescriptions[0].State = commonpb.IndexState_Finished
				flag = true
			}

			s, err := SuccessStatus()
			resp.Status = s
			return resp, err
		})

		assert.Nil(t, c.CreateIndex(ctx, testCollectionName, fieldName, idx, false, WithIndexName("test-index")))
		assert.True(t, flag)

		mockServer.DelInjection(MDescribeIndex)
	})
}

func TestGrpcClientDropIndex(t *testing.T) {
	ctx := context.Background()
	c := testClient(ctx, t)
	mockServer.SetInjection(MHasCollection, hasCollectionDefault)
	mockServer.SetInjection(MDescribeCollection, describeCollectionInjection(t, 0, testCollectionName, defaultSchema()))
	assert.Nil(t, c.DropIndex(ctx, testCollectionName, "vector", WithIndexMsgBase(&commonpb.MsgBase{})))
}

func TestGrpcClientDescribeIndex(t *testing.T) {
	ctx := context.Background()
	mockServer.SetInjection(MHasCollection, hasCollectionDefault)
	mockServer.SetInjection(MDescribeCollection, describeCollectionInjection(t, 0, testCollectionName, defaultSchema()))

	c := testClient(ctx, t)

	fieldName := "vector"

	t.Run("normal describe index", func(t *testing.T) {
		mockServer.SetInjection(MDescribeIndex, func(_ context.Context, raw proto.Message) (proto.Message, error) {
			req, ok := raw.(*milvuspb.DescribeIndexRequest)
			resp := &milvuspb.DescribeIndexResponse{}
			if !ok {
				s, err := BadRequestStatus()
				resp.Status = s
				return resp, err
			}
			assert.Equal(t, fieldName, req.GetFieldName())
			assert.Equal(t, testCollectionName, req.GetCollectionName())
			resp.IndexDescriptions = []*milvuspb.IndexDescription{
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
		defer mockServer.DelInjection(MDescribeIndex)
		mockServer.SetInjection(MDescribeIndex, func(_ context.Context, raw proto.Message) (proto.Message, error) {
			resp := &milvuspb.DescribeIndexResponse{}

			return resp, errors.New("mockServer.d error")
		})

		_, err := c.DescribeIndex(ctx, testCollectionName, fieldName)
		assert.Error(t, err)

		mockServer.SetInjection(MDescribeIndex, func(_ context.Context, raw proto.Message) (proto.Message, error) {
			resp := &milvuspb.DescribeIndexResponse{}
			resp.Status = &commonpb.Status{ErrorCode: commonpb.ErrorCode_UnexpectedError}
			return resp, nil
		})

		_, err = c.DescribeIndex(ctx, testCollectionName, fieldName)
		assert.Error(t, err)
	})
}

func TestGrpcGetIndexBuildProgress(t *testing.T) {
	ctx := context.Background()
	mockServer.SetInjection(MHasCollection, hasCollectionDefault)
	mockServer.SetInjection(MDescribeCollection, describeCollectionInjection(t, 0, testCollectionName, defaultSchema()))

	tc := testClient(ctx, t)
	c := tc.(*GrpcClient) // since GetIndexBuildProgress is not exposed

	t.Run("normal get index build progress", func(t *testing.T) {
		var total, built int64

		mockServer.SetInjection(MGetIndexBuildProgress, func(_ context.Context, raw proto.Message) (proto.Message, error) {
			req, ok := raw.(*milvuspb.GetIndexBuildProgressRequest)
			if !ok {
				t.FailNow()
			}
			assert.Equal(t, testCollectionName, req.GetCollectionName())
			resp := &milvuspb.GetIndexBuildProgressResponse{
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
		defer mockServer.DelInjection(MGetIndexBuildProgress)
		mockServer.SetInjection(MGetIndexBuildProgress, func(_ context.Context, raw proto.Message) (proto.Message, error) {
			_, ok := raw.(*milvuspb.GetIndexBuildProgressRequest)
			if !ok {
				t.FailNow()
			}
			resp := &milvuspb.GetIndexBuildProgressResponse{}
			return resp, errors.New("mockServer.d error")
		})

		_, _, err := c.GetIndexBuildProgress(ctx, testCollectionName, testVectorField)
		assert.Error(t, err)

		mockServer.SetInjection(MGetIndexBuildProgress, func(_ context.Context, raw proto.Message) (proto.Message, error) {
			_, ok := raw.(*milvuspb.GetIndexBuildProgressRequest)
			if !ok {
				t.FailNow()
			}
			resp := &milvuspb.GetIndexBuildProgressResponse{}
			resp.Status = &commonpb.Status{
				ErrorCode: commonpb.ErrorCode_UnexpectedError,
			}
			return resp, nil
		})
		_, _, err = c.GetIndexBuildProgress(ctx, testCollectionName, testVectorField)
		assert.Error(t, err)
	})

}
