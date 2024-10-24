package client

import (
	"context"
	"fmt"
	"math/rand"
	"testing"
	"time"

	"github.com/cockroachdb/errors"

	"github.com/golang/protobuf/proto"
	"github.com/milvus-io/milvus-proto/go-api/v2/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v2/milvuspb"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/suite"
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

type IndexSuite struct {
	MockSuiteBase
}

func (s *IndexSuite) TestGetIndexBuildProgress() {
	c := s.client
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	collectionName := fmt.Sprintf("coll_%s", randStr(6))
	fieldName := fmt.Sprintf("field_%d", rand.Int31n(10))
	indexName := fmt.Sprintf("index_%s", randStr(4))

	s.Run("normal_case", func() {
		totalRows := rand.Int63n(10000)
		indexedRows := rand.Int63n(totalRows)

		defer s.resetMock()

		s.mock.EXPECT().DescribeIndex(mock.Anything, mock.Anything).RunAndReturn(func(ctx context.Context, dir *milvuspb.DescribeIndexRequest) (*milvuspb.DescribeIndexResponse, error) {
			s.Equal(collectionName, dir.GetCollectionName())
			s.Equal(fieldName, dir.GetFieldName())
			s.Equal(indexName, dir.GetIndexName())
			return &milvuspb.DescribeIndexResponse{
				Status: s.getSuccessStatus(),
				IndexDescriptions: []*milvuspb.IndexDescription{
					{
						IndexName:   indexName,
						TotalRows:   totalRows,
						IndexedRows: indexedRows,
						State:       commonpb.IndexState_InProgress,
					},
				},
			}, nil
		}).Once()

		totalResult, indexedResult, err := c.GetIndexBuildProgress(ctx, collectionName, fieldName, WithIndexName(indexName))
		s.NoError(err)
		s.Equal(totalRows, totalResult)
		s.Equal(indexedRows, indexedResult)
	})

	s.Run("index_not_found", func() {
		defer s.resetMock()

		s.mock.EXPECT().DescribeIndex(mock.Anything, mock.Anything).RunAndReturn(func(ctx context.Context, dir *milvuspb.DescribeIndexRequest) (*milvuspb.DescribeIndexResponse, error) {
			s.Equal(collectionName, dir.GetCollectionName())
			s.Equal(fieldName, dir.GetFieldName())
			s.Equal(indexName, dir.GetIndexName())
			return &milvuspb.DescribeIndexResponse{
				Status:            s.getSuccessStatus(),
				IndexDescriptions: []*milvuspb.IndexDescription{},
			}, nil
		}).Once()

		_, _, err := c.GetIndexBuildProgress(ctx, collectionName, fieldName, WithIndexName(indexName))
		s.Error(err)
	})

	s.Run("build_failed", func() {
		defer s.resetMock()

		s.mock.EXPECT().DescribeIndex(mock.Anything, mock.Anything).RunAndReturn(func(ctx context.Context, dir *milvuspb.DescribeIndexRequest) (*milvuspb.DescribeIndexResponse, error) {
			s.Equal(collectionName, dir.GetCollectionName())
			s.Equal(fieldName, dir.GetFieldName())
			s.Equal(indexName, dir.GetIndexName())
			return &milvuspb.DescribeIndexResponse{
				Status: s.getSuccessStatus(),
				IndexDescriptions: []*milvuspb.IndexDescription{
					{
						IndexName: indexName,
						State:     commonpb.IndexState_Failed,
					},
				},
			}, nil
		}).Once()

		_, _, err := c.GetIndexBuildProgress(ctx, collectionName, fieldName, WithIndexName(indexName))
		s.Error(err)
	})

	s.Run("server_error", func() {
		defer s.resetMock()

		s.mock.EXPECT().DescribeIndex(mock.Anything, mock.Anything).RunAndReturn(func(ctx context.Context, dir *milvuspb.DescribeIndexRequest) (*milvuspb.DescribeIndexResponse, error) {
			s.Equal(collectionName, dir.GetCollectionName())
			s.Equal(fieldName, dir.GetFieldName())
			s.Equal(indexName, dir.GetIndexName())
			return nil, errors.New("mocked")
		}).Once()

		_, _, err := c.GetIndexBuildProgress(ctx, collectionName, fieldName, WithIndexName(indexName))
		s.Error(err)
	})
}

func TestIndex(t *testing.T) {
	suite.Run(t, new(IndexSuite))
}
