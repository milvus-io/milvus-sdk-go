package client

import (
	"context"
	"errors"
	"fmt"
	"math/rand"
	"testing"

	"github.com/golang/protobuf/proto"
	common "github.com/milvus-io/milvus-proto/go-api/v2/commonpb"
	server "github.com/milvus-io/milvus-proto/go-api/v2/milvuspb"
	"github.com/milvus-io/milvus-proto/go-api/v2/schemapb"
	schema "github.com/milvus-io/milvus-proto/go-api/v2/schemapb"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/suite"
)

// default HasCollection injection, returns true only when collection name is `testCollectionName`
var hasCollectionDefault = func(_ context.Context, raw proto.Message) (proto.Message, error) {
	req, ok := raw.(*server.HasCollectionRequest)
	resp := &server.BoolResponse{}
	if !ok {
		s, err := BadRequestStatus()
		resp.Status = s
		return s, err
	}
	resp.Value = req.GetCollectionName() == testCollectionName
	s, err := SuccessStatus()
	resp.Status = s
	return resp, err
}

func describeCollectionInjection(_ *testing.T, collID int64, collName string, sch *entity.Schema) func(_ context.Context, raw proto.Message) (proto.Message, error) {
	return func(_ context.Context, raw proto.Message) (proto.Message, error) {
		_, ok := raw.(*server.DescribeCollectionRequest)
		resp := &server.DescribeCollectionResponse{}
		if !ok {
			s, err := BadRequestStatus()
			resp.Status = s
			return resp, err
		}

		sch := sch
		resp.Schema = sch.ProtoMessage()
		resp.CollectionID = collID
		resp.CollectionName = collName

		s, err := SuccessStatus()
		resp.Status = s

		return resp, err
	}
}

func TestGrpcClientDescribeCollection(t *testing.T) {
	ctx := context.Background()

	c := testClient(ctx, t)

	collectionID := rand.Int63()

	mockServer.SetInjection(MDescribeCollection, describeCollectionInjection(t, collectionID, testCollectionName, defaultSchema()))

	collection, err := c.DescribeCollection(ctx, testCollectionName)
	assert.Nil(t, err)
	if assert.NotNil(t, collection) {
		assert.Equal(t, collectionID, collection.ID)
	}
}

func TestGrpcClientGetCollectionStatistics(t *testing.T) {
	ctx := context.Background()

	c := testClient(ctx, t)

	stat := make(map[string]string)
	stat["row_count"] = "0"

	mockServer.SetInjection(MGetCollectionStatistics, func(_ context.Context, raw proto.Message) (proto.Message, error) {
		req, ok := raw.(*server.GetCollectionStatisticsRequest)
		resp := &server.GetCollectionStatisticsResponse{}
		if !ok {
			s, err := BadRequestStatus()
			resp.Status = s
			return resp, err
		}
		assert.Equal(t, testCollectionName, req.GetCollectionName())
		s, err := SuccessStatus()
		resp.Status, resp.Stats = s, entity.MapKvPairs(stat)
		return resp, err
	})

	rStat, err := c.GetCollectionStatistics(ctx, testCollectionName)
	assert.Nil(t, err)
	if assert.NotNil(t, rStat) {
		for k, v := range stat {
			rv, has := rStat[k]
			assert.True(t, has)
			assert.Equal(t, v, rv)
		}
	}
}

func TestGrpcClientGetLoadingProgress(t *testing.T) {
	ctx := context.Background()
	c := testClient(ctx, t)

	mockServer.SetInjection(MHasCollection, hasCollectionDefault)

	mockServer.SetInjection(MGetLoadingProgress, func(_ context.Context, raw proto.Message) (proto.Message, error) {
		req, ok := raw.(*server.GetLoadingProgressRequest)
		if !ok {
			return BadRequestStatus()
		}
		resp := &server.GetLoadingProgressResponse{}
		if !ok {
			s, err := BadRequestStatus()
			resp.Status = s
			return resp, err
		}
		assert.Equal(t, testCollectionName, req.GetCollectionName())
		s, err := SuccessStatus()
		resp.Status, resp.Progress = s, 100
		return resp, err
	})

	progress, err := c.GetLoadingProgress(ctx, testCollectionName, []string{})
	assert.NoError(t, err)
	assert.Equal(t, int64(100), progress)
}

func TestGrpcClientGetLoadState(t *testing.T) {

	ctx := context.Background()
	c := testClient(ctx, t)

	mockServer.SetInjection(MHasCollection, hasCollectionDefault)

	mockServer.SetInjection(MGetLoadState, func(_ context.Context, raw proto.Message) (proto.Message, error) {
		req, ok := raw.(*server.GetLoadStateRequest)
		if !ok {
			return BadRequestStatus()
		}
		resp := &server.GetLoadStateResponse{}
		if !ok {
			s, err := BadRequestStatus()
			resp.Status = s
			return resp, err
		}
		assert.Equal(t, testCollectionName, req.GetCollectionName())
		s, err := SuccessStatus()
		resp.Status, resp.State = s, common.LoadState_LoadStateLoaded
		return resp, err
	})

	state, err := c.GetLoadState(ctx, testCollectionName, []string{})
	assert.NoError(t, err)
	assert.Equal(t, entity.LoadStateLoaded, state)
}

type CollectionSuite struct {
	MockSuiteBase
}

func (s *CollectionSuite) TestListCollections() {
	c := s.client
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	type testCase struct {
		ids     []int64
		names   []string
		collNum int
		inMem   []int64
	}
	caseLen := 5
	cases := make([]testCase, 0, caseLen)
	for i := 0; i < caseLen; i++ {
		collNum := rand.Intn(5) + 2
		tc := testCase{
			ids:     make([]int64, 0, collNum),
			names:   make([]string, 0, collNum),
			collNum: collNum,
		}
		base := rand.Intn(1000)
		for j := 0; j < collNum; j++ {
			base += rand.Intn(1000)
			tc.ids = append(tc.ids, int64(base))
			base += rand.Intn(500)
			tc.names = append(tc.names, fmt.Sprintf("coll_%d", base))
			inMem := rand.Intn(100)
			if inMem%2 == 0 {

				tc.inMem = append(tc.inMem, 100)
			} else {
				tc.inMem = append(tc.inMem, 0)
			}
		}
		cases = append(cases, tc)
	}

	for i, tc := range cases {
		s.Run(fmt.Sprintf("run_%d", i), func() {
			s.resetMock()
			s.mock.EXPECT().ShowCollections(mock.Anything, mock.AnythingOfType("*milvuspb.ShowCollectionsRequest")).
				Return(&server.ShowCollectionsResponse{
					Status:              &common.Status{},
					CollectionIds:       tc.ids,
					CollectionNames:     tc.names,
					InMemoryPercentages: tc.inMem,
				}, nil)

			collections, err := c.ListCollections(ctx)

			s.Require().Equal(tc.collNum, len(collections))
			s.Require().NoError(err)

			// assert element match
			rids := make([]int64, 0, len(collections))
			rnames := make([]string, 0, len(collections))
			for _, collection := range collections {
				rids = append(rids, collection.ID)
				rnames = append(rnames, collection.Name)
			}

			s.ElementsMatch(tc.ids, rids)
			s.ElementsMatch(tc.names, rnames)
			// assert id & name match
			for idx, rid := range rids {
				for jdx, id := range tc.ids {
					if rid == id {
						s.Equal(tc.names[idx], rnames[idx])
						s.Equal(tc.inMem[jdx] == 100, collections[idx].Loaded)
					}
				}
			}
		})
	}
}

func (s *CollectionSuite) TestCreateCollection() {
	c := s.client
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	s.Run("normal_creation", func() {
		ds := defaultSchema()
		shardsNum := int32(1)

		defer s.resetMock()
		s.mock.EXPECT().CreateCollection(mock.Anything, mock.AnythingOfType("*milvuspb.CreateCollectionRequest")).
			Run(func(ctx context.Context, req *server.CreateCollectionRequest) {
				s.Equal(testCollectionName, req.GetCollectionName())
				sschema := &schema.CollectionSchema{}
				s.Require().NoError(proto.Unmarshal(req.GetSchema(), sschema))
				s.Require().Equal(len(ds.Fields), len(sschema.Fields))
				for idx, fieldSchema := range ds.Fields {
					s.Equal(fieldSchema.Name, sschema.GetFields()[idx].GetName())
					s.Equal(fieldSchema.PrimaryKey, sschema.GetFields()[idx].GetIsPrimaryKey())
					s.Equal(fieldSchema.AutoID, sschema.GetFields()[idx].GetAutoID())
					s.EqualValues(fieldSchema.DataType, sschema.GetFields()[idx].GetDataType())
				}
				s.Equal(shardsNum, req.GetShardsNum())
				s.Equal(common.ConsistencyLevel_Bounded, req.GetConsistencyLevel())
			}).
			Return(&common.Status{ErrorCode: common.ErrorCode_Success}, nil)

		err := c.CreateCollection(ctx, ds, shardsNum)
		s.NoError(err)
	})

	s.Run("create_with_consistency_level", func() {
		ds := defaultSchema()
		shardsNum := int32(1)
		defer s.resetMock()
		s.mock.EXPECT().CreateCollection(mock.Anything, mock.AnythingOfType("*milvuspb.CreateCollectionRequest")).
			Run(func(ctx context.Context, req *server.CreateCollectionRequest) {
				s.Equal(testCollectionName, req.GetCollectionName())
				sschema := &schema.CollectionSchema{}
				s.Require().NoError(proto.Unmarshal(req.GetSchema(), sschema))
				s.Require().Equal(len(ds.Fields), len(sschema.Fields))
				for idx, fieldSchema := range ds.Fields {
					s.Equal(fieldSchema.Name, sschema.GetFields()[idx].GetName())
					s.Equal(fieldSchema.PrimaryKey, sschema.GetFields()[idx].GetIsPrimaryKey())
					s.Equal(fieldSchema.AutoID, sschema.GetFields()[idx].GetAutoID())
					s.EqualValues(fieldSchema.DataType, sschema.GetFields()[idx].GetDataType())
				}
				s.Equal(shardsNum, req.GetShardsNum())
				s.Equal(common.ConsistencyLevel_Eventually, req.GetConsistencyLevel())

			}).
			Return(&common.Status{ErrorCode: common.ErrorCode_Success}, nil)

		err := c.CreateCollection(ctx, ds, shardsNum, WithConsistencyLevel(entity.ClEventually))
		s.NoError(err)
	})

	s.Run("invalid_schemas", func() {

		type testCase struct {
			name   string
			schema *entity.Schema
		}
		cases := []testCase{
			{
				name:   "empty_fields",
				schema: entity.NewSchema().WithName(testCollectionName),
			},
			{
				name: "empty_collection_name",
				schema: entity.NewSchema().
					WithField(entity.NewField().WithName("int64").WithDataType(entity.FieldTypeInt64).WithIsPrimaryKey(true)).
					WithField(entity.NewField().WithName("vector").WithDataType(entity.FieldTypeFloatVector).WithDim(128)),
			},
			{
				name: "multiple_primary_key",
				schema: entity.NewSchema().WithName(testCollectionName).
					WithField(entity.NewField().WithName("int64").WithDataType(entity.FieldTypeInt64).WithIsPrimaryKey(true)).
					WithField(entity.NewField().WithName("int64_2").WithDataType(entity.FieldTypeInt64).WithIsPrimaryKey(true)).
					WithField(entity.NewField().WithName("vector").WithDataType(entity.FieldTypeFloatVector).WithDim(128)),
			},
			{
				name: "multiple_auto_id",
				schema: entity.NewSchema().WithName(testCollectionName).
					WithField(entity.NewField().WithName("int64").WithDataType(entity.FieldTypeInt64).WithIsPrimaryKey(true).WithIsAutoID(true)).
					WithField(entity.NewField().WithName("int64_2").WithDataType(entity.FieldTypeInt64).WithIsAutoID(true)).
					WithField(entity.NewField().WithName("vector").WithDataType(entity.FieldTypeFloatVector).WithDim(128)),
			},
			{
				name: "bad_pk_type",
				schema: entity.NewSchema().
					WithField(entity.NewField().WithName("int64").WithDataType(entity.FieldTypeDouble).WithIsPrimaryKey(true)).
					WithField(entity.NewField().WithName("vector").WithDataType(entity.FieldTypeFloatVector).WithDim(128)),
			},
		}

		for _, tc := range cases {
			s.Run(tc.name, func() {
				err := c.CreateCollection(ctx, tc.schema, 1)
				s.Error(err)
			})
		}
	})

	s.Run("server_returns_error", func() {
		s.Run("create_collection_error", func() {
			defer s.resetMock()
			s.mock.EXPECT().CreateCollection(mock.Anything, mock.AnythingOfType("*milvuspb.CreateCollectionRequest")).
				Return(nil, errors.New("mocked grpc error"))

			err := c.CreateCollection(ctx, defaultSchema(), 1)
			s.Error(err)
		})

		s.Run("create_collection_fail", func() {
			defer s.resetMock()
			s.mock.EXPECT().CreateCollection(mock.Anything, mock.AnythingOfType("*milvuspb.CreateCollectionRequest")).
				Return(&common.Status{ErrorCode: common.ErrorCode_UnexpectedError}, nil)

			err := c.CreateCollection(ctx, defaultSchema(), 1)
			s.Error(err)
		})
	})

	s.Run("feature_not_support", func() {
		cases := []struct {
			tag  string
			flag uint64
		}{
			{tag: "json", flag: disableJSON},
			{tag: "partition_key", flag: disableParitionKey},
			{tag: "dyanmic_schema", flag: disableDynamicSchema},
		}
		sch := entity.NewSchema().WithName("all_feature").WithDynamicFieldEnabled(true).
			WithField(entity.NewField().WithName("id").WithDataType(entity.FieldTypeInt64).WithIsPrimaryKey(true)).
			WithField(entity.NewField().WithName("embedding").WithDataType(entity.FieldTypeFloatVector).WithDim(128)).
			WithField(entity.NewField().WithName("partition").WithDataType(entity.FieldTypeInt64).WithIsPartitionKey(true)).
			WithField(entity.NewField().WithName("dynamic").WithDataType(entity.FieldTypeJSON).WithIsDynamic(true))
		for _, tc := range cases {
			s.Run(tc.tag, func() {
				grpcClient, ok := c.(*GrpcClient)
				s.Require().True(ok)
				grpcClient.config.addFlags(tc.flag)
				defer grpcClient.config.resetFlags(tc.flag)

				err := c.CreateCollection(ctx, sch, 1)
				s.Error(err)
				s.ErrorIs(err, ErrFeatureNotSupported)
			})
		}
	})
}

func (s *CollectionSuite) TestNewCollection() {
	c := s.client
	ctx, cancel := context.WithCancel(context.Background())

	defer cancel()
	s.resetMock()

	s.Run("all_default", func() {
		defer s.resetMock()

		created := false
		s.mock.EXPECT().CreateCollection(mock.Anything, mock.AnythingOfType("*milvuspb.CreateCollectionRequest")).
			Run(func(ctx context.Context, req *server.CreateCollectionRequest) {
				s.Equal(testCollectionName, req.GetCollectionName())
				sschema := &schema.CollectionSchema{}
				s.Require().NoError(proto.Unmarshal(req.GetSchema(), sschema))
				s.Require().Equal(2, len(sschema.Fields))
				for _, field := range sschema.Fields {
					if field.GetName() == "id" {
						s.Equal(schemapb.DataType_Int64, field.GetDataType())
					}
					if field.GetName() == "vector" {
						s.Equal(schemapb.DataType_FloatVector, field.GetDataType())
					}
				}

				s.Equal(entity.DefaultShardNumber, req.GetShardsNum())
				s.Equal(entity.DefaultConsistencyLevel.CommonConsistencyLevel(), req.GetConsistencyLevel())
				created = true
			}).
			Return(&common.Status{ErrorCode: common.ErrorCode_Success}, nil)

		s.mock.EXPECT().CreateIndex(mock.Anything, mock.Anything).Return(&common.Status{ErrorCode: common.ErrorCode_Success}, nil)
		s.mock.EXPECT().Flush(mock.Anything, mock.Anything).Return(&server.FlushResponse{
			Status:     &common.Status{ErrorCode: common.ErrorCode_Success},
			CollSegIDs: map[string]*schema.LongArray{},
		}, nil)
		s.mock.EXPECT().DescribeIndex(mock.Anything, mock.Anything).Return(&server.DescribeIndexResponse{
			Status: &common.Status{ErrorCode: common.ErrorCode_Success},
			IndexDescriptions: []*server.IndexDescription{
				{FieldName: "vector", State: common.IndexState_Finished},
			},
		}, nil)
		s.mock.EXPECT().LoadCollection(mock.Anything, mock.Anything).Return(&common.Status{ErrorCode: common.ErrorCode_Success}, nil)
		s.mock.EXPECT().GetLoadingProgress(mock.Anything, mock.Anything).Return(&server.GetLoadingProgressResponse{
			Status:   &common.Status{ErrorCode: common.ErrorCode_Success},
			Progress: 100,
		}, nil)
		s.mock.EXPECT().DescribeCollection(mock.Anything, mock.Anything).Call.Return(func(_ context.Context, _ *server.DescribeCollectionRequest) *server.DescribeCollectionResponse {
			if created {
				return &server.DescribeCollectionResponse{
					Status: &common.Status{},
					Schema: &schema.CollectionSchema{
						Fields: []*schema.FieldSchema{
							{Name: "id", DataType: schema.DataType_VarChar},
							{Name: "vector", DataType: schema.DataType_FloatVector},
						},
					},
				}
			}
			return &server.DescribeCollectionResponse{Status: &common.Status{ErrorCode: common.ErrorCode_CollectionNotExists}}
		}, nil)

		err := c.NewCollection(ctx, testCollectionName, testVectorDim)
		s.NoError(err)
	})

	s.Run("with_custom_set", func() {
		defer s.resetMock()
		created := false
		s.mock.EXPECT().CreateCollection(mock.Anything, mock.AnythingOfType("*milvuspb.CreateCollectionRequest")).
			Run(func(ctx context.Context, req *server.CreateCollectionRequest) {
				s.Equal(testCollectionName, req.GetCollectionName())
				sschema := &schema.CollectionSchema{}
				s.Require().NoError(proto.Unmarshal(req.GetSchema(), sschema))
				s.Require().Equal(2, len(sschema.Fields))
				for _, field := range sschema.Fields {
					if field.GetName() == "my_pk" {
						s.Equal(schemapb.DataType_VarChar, field.GetDataType())
					}
					if field.GetName() == "embedding" {
						s.Equal(schemapb.DataType_FloatVector, field.GetDataType())
					}
				}

				s.Equal(entity.DefaultShardNumber, req.GetShardsNum())
				s.Equal(entity.ClEventually.CommonConsistencyLevel(), req.GetConsistencyLevel())
				created = true
			}).
			Return(&common.Status{ErrorCode: common.ErrorCode_Success}, nil)
		s.mock.EXPECT().CreateIndex(mock.Anything, mock.Anything).Return(&common.Status{ErrorCode: common.ErrorCode_Success}, nil)
		s.mock.EXPECT().Flush(mock.Anything, mock.Anything).Return(&server.FlushResponse{
			Status:     &common.Status{ErrorCode: common.ErrorCode_Success},
			CollSegIDs: map[string]*schema.LongArray{},
		}, nil)
		s.mock.EXPECT().DescribeIndex(mock.Anything, mock.Anything).Return(&server.DescribeIndexResponse{
			Status: &common.Status{ErrorCode: common.ErrorCode_Success},
			IndexDescriptions: []*server.IndexDescription{
				{FieldName: "embedding", State: common.IndexState_Finished},
			},
		}, nil)
		s.mock.EXPECT().LoadCollection(mock.Anything, mock.Anything).Return(&common.Status{ErrorCode: common.ErrorCode_Success}, nil)
		s.mock.EXPECT().GetLoadingProgress(mock.Anything, mock.Anything).Return(&server.GetLoadingProgressResponse{
			Status:   &common.Status{ErrorCode: common.ErrorCode_Success},
			Progress: 100,
		}, nil)
		s.mock.EXPECT().DescribeCollection(mock.Anything, mock.Anything).Call.Return(func(_ context.Context, _ *server.DescribeCollectionRequest) *server.DescribeCollectionResponse {
			if created {
				return &server.DescribeCollectionResponse{
					Status: &common.Status{},
					Schema: &schema.CollectionSchema{
						Fields: []*schema.FieldSchema{
							{Name: "my_pk", DataType: schema.DataType_VarChar},
							{Name: "embedding", DataType: schema.DataType_FloatVector},
						},
					},
				}
			}
			return &server.DescribeCollectionResponse{Status: &common.Status{ErrorCode: common.ErrorCode_CollectionNotExists}}
		}, nil)

		err := c.NewCollection(ctx, testCollectionName, testVectorDim, WithPKFieldName("my_pk"), WithPKFieldType(entity.FieldTypeVarChar), WithVectorFieldName("embedding"), WithConsistencyLevel(entity.ClEventually))
		s.NoError(err)
	})

	s.Run("varchar_autoid", func() {
		defer s.resetMock()

		err := c.NewCollection(ctx, testCollectionName, testVectorDim, WithPKFieldType(entity.FieldTypeVarChar), WithAutoID(true))
		s.Error(err)
	})
}

func (s *CollectionSuite) TestAlterCollection() {
	c := s.client
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	s.Run("normal_run", func() {
		defer s.resetMock()

		s.setupDescribeCollection(testCollectionName, nil)
		s.mock.EXPECT().AlterCollection(mock.Anything, mock.AnythingOfType("*milvuspb.AlterCollectionRequest")).
			Return(&common.Status{}, nil)

		err := c.AlterCollection(ctx, testCollectionName, entity.CollectionTTL(100000))
		s.NoError(err)
	})

	s.Run("collection_not_exist", func() {
		defer s.resetMock()

		s.setupDescribeCollectionError(common.ErrorCode_CollectionNotExists, nil)

		err := c.AlterCollection(ctx, testCollectionName, entity.CollectionTTL(100000))
		s.Error(err)
	})

	s.Run("no_attributes", func() {
		defer s.resetMock()

		s.setupDescribeCollection(testCollectionName, nil)
		err := c.AlterCollection(ctx, testCollectionName)
		s.Error(err)
	})

	s.Run("request_fails", func() {
		defer s.resetMock()

		s.setupDescribeCollection(testCollectionName, nil)
		s.mock.EXPECT().AlterCollection(mock.Anything, mock.AnythingOfType("*milvuspb.AlterCollectionRequest")).
			Return(nil, errors.New("mocked"))

		err := c.AlterCollection(ctx, testCollectionName, entity.CollectionTTL(100000))
		s.Error(err)
	})

	s.Run("server_return_error", func() {
		defer s.resetMock()

		s.setupDescribeCollection(testCollectionName, nil)
		s.mock.EXPECT().AlterCollection(mock.Anything, mock.AnythingOfType("*milvuspb.AlterCollectionRequest")).
			Return(&common.Status{ErrorCode: common.ErrorCode_UnexpectedError}, nil)

		err := c.AlterCollection(ctx, testCollectionName, entity.CollectionTTL(100000))
		s.Error(err)
	})

	s.Run("service_not_ready", func() {
		c := &GrpcClient{}
		err := c.AlterCollection(ctx, testCollectionName, entity.CollectionTTL(100000))
		s.ErrorIs(err, ErrClientNotReady)
	})
}

func (s *CollectionSuite) TestLoadCollection() {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	c := s.client

	s.Run("normal_run_async", func() {
		defer s.resetMock()
		s.setupDescribeCollection(testCollectionName, nil)

		s.mock.EXPECT().LoadCollection(mock.Anything, mock.Anything).Return(&common.Status{ErrorCode: common.ErrorCode_Success}, nil)

		err := c.LoadCollection(ctx, testCollectionName, true)
		s.NoError(err)
	})

	s.Run("normal_run_sync", func() {
		defer s.resetMock()
		s.setupDescribeCollection(testCollectionName, nil)

		s.mock.EXPECT().LoadCollection(mock.Anything, mock.Anything).Return(&common.Status{ErrorCode: common.ErrorCode_Success}, nil)
		s.mock.EXPECT().GetLoadingProgress(mock.Anything, mock.Anything).
			Return(&server.GetLoadingProgressResponse{
				Status:   &common.Status{ErrorCode: common.ErrorCode_Success},
				Progress: 100,
			}, nil)

		err := c.LoadCollection(ctx, testCollectionName, false)
		s.NoError(err)
	})

	s.Run("load_default_replica", func() {
		defer s.resetMock()
		s.setupDescribeCollection(testCollectionName, nil)

		s.mock.EXPECT().LoadCollection(mock.Anything, mock.Anything).Run(func(_ context.Context, req *server.LoadCollectionRequest) {
			s.Equal(testDefaultReplicaNumber, req.GetReplicaNumber())
			s.Equal(testCollectionName, req.GetCollectionName())
		}).Return(&common.Status{ErrorCode: common.ErrorCode_Success}, nil)

		err := c.LoadCollection(ctx, testCollectionName, true)
		s.NoError(err)
	})

	s.Run("load_multiple_replica", func() {
		defer s.resetMock()
		s.setupDescribeCollection(testCollectionName, nil)

		s.mock.EXPECT().LoadCollection(mock.Anything, mock.Anything).Run(func(_ context.Context, req *server.LoadCollectionRequest) {
			s.Equal(testMultiReplicaNumber, req.GetReplicaNumber())
			s.Equal(testCollectionName, req.GetCollectionName())
		}).Return(&common.Status{ErrorCode: common.ErrorCode_Success}, nil)

		err := c.LoadCollection(ctx, testCollectionName, true, WithReplicaNumber(testMultiReplicaNumber))
		s.NoError(err)
	})

	s.Run("has_collection_failure", func() {
		s.Run("return_false", func() {
			defer s.resetMock()
			s.setupDescribeCollectionError(common.ErrorCode_CollectionNotExists, nil)

			err := c.LoadCollection(ctx, testCollectionName, true)
			s.Error(err)
		})

		s.Run("return_error", func() {
			defer s.resetMock()
			s.setupDescribeCollectionError(common.ErrorCode_UnexpectedError, errors.New("mocked error"))

			err := c.LoadCollection(ctx, testCollectionName, true)
			s.Error(err)
		})
	})

	s.Run("load_collection_failure", func() {
		s.Run("failure_status", func() {
			defer s.resetMock()
			s.setupDescribeCollection(testCollectionName, nil)

			s.mock.EXPECT().LoadCollection(mock.Anything, mock.Anything).
				Return(&common.Status{ErrorCode: common.ErrorCode_UnexpectedError}, nil)

			err := c.LoadCollection(ctx, testCollectionName, true)
			s.Error(err)
		})

		s.Run("return_error", func() {
			defer s.resetMock()
			s.setupDescribeCollection(testCollectionName, nil)

			s.mock.EXPECT().LoadCollection(mock.Anything, mock.Anything).
				Return(nil, errors.New("mock error"))

			err := c.LoadCollection(ctx, testCollectionName, true)
			s.Error(err)
		})
	})

	s.Run("get_loading_progress_failure", func() {
		defer s.resetMock()
		s.setupDescribeCollection(testCollectionName, nil)

		s.mock.EXPECT().LoadCollection(mock.Anything, mock.Anything).Return(&common.Status{ErrorCode: common.ErrorCode_Success}, nil)
		s.mock.EXPECT().GetLoadingProgress(mock.Anything, mock.Anything).
			Return(nil, errors.New("mock error"))

		err := c.LoadCollection(ctx, testCollectionName, false)
		s.Error(err)
	})

	s.Run("service_not_ready", func() {
		c := &GrpcClient{}
		err := c.LoadCollection(ctx, testCollectionName, false)
		s.ErrorIs(err, ErrClientNotReady)
	})
}

func (s *CollectionSuite) TestDropCollection() {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	c := s.client

	s.Run("normal_run", func() {
		defer s.resetMock()

		s.setupDescribeCollection(testCollectionName, nil)
		s.mock.EXPECT().DropCollection(mock.Anything, mock.Anything).Run(func(_ context.Context, req *server.DropCollectionRequest) {
			s.Equal(testCollectionName, req.GetCollectionName())
		}).Return(&common.Status{}, nil)

		err := c.DropCollection(ctx, testCollectionName)
		s.NoError(err)
	})

	s.Run("coll_not_exists", func() {
		defer s.resetMock()

		s.setupDescribeCollectionError(common.ErrorCode_CollectionNotExists, nil)

		err := c.DropCollection(ctx, testCollectionName)
		s.Error(err)
	})

	s.Run("drop_collection_fail", func() {
		s.Run("return_error", func() {
			defer s.resetMock()

			s.setupDescribeCollection(testCollectionName, nil)
			s.mock.EXPECT().DropCollection(mock.Anything, mock.Anything).Return(nil, errors.New("mocked"))

			err := c.DropCollection(ctx, testCollectionName)
			s.Error(err)
		})

		s.Run("return_error_status", func() {
			defer s.resetMock()

			s.setupDescribeCollection(testCollectionName, nil)
			s.mock.EXPECT().DropCollection(mock.Anything, mock.Anything).
				Return(&common.Status{ErrorCode: common.ErrorCode_UnexpectedError}, nil)

			err := c.DropCollection(ctx, testCollectionName)
			s.Error(err)
		})
	})

	s.Run("service_not_ready", func() {
		c := &GrpcClient{}
		err := c.DropCollection(ctx, testCollectionName)
		s.ErrorIs(err, ErrClientNotReady)
	})
}

func (s *CollectionSuite) TestReleaseCollection() {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	c := s.client

	s.Run("normal_run", func() {
		defer s.resetMock()

		s.setupDescribeCollection(testCollectionName, nil)
		s.mock.EXPECT().ReleaseCollection(mock.Anything, mock.Anything).Run(func(_ context.Context, req *server.ReleaseCollectionRequest) {
			s.Equal(testCollectionName, req.GetCollectionName())
		}).Return(&common.Status{}, nil)

		err := c.ReleaseCollection(ctx, testCollectionName)
		s.NoError(err)
	})

	s.Run("coll_not_exists", func() {
		defer s.resetMock()

		s.setupDescribeCollectionError(common.ErrorCode_CollectionNotExists, nil)

		err := c.ReleaseCollection(ctx, testCollectionName)
		s.Error(err)
	})

	s.Run("release_collection_fail", func() {
		s.Run("return_error", func() {
			defer s.resetMock()

			s.setupDescribeCollection(testCollectionName, nil)
			s.mock.EXPECT().ReleaseCollection(mock.Anything, mock.Anything).Return(nil, errors.New("mocked"))

			err := c.ReleaseCollection(ctx, testCollectionName)
			s.Error(err)
		})

		s.Run("return_error_status", func() {
			defer s.resetMock()

			s.setupDescribeCollection(testCollectionName, nil)
			s.mock.EXPECT().ReleaseCollection(mock.Anything, mock.Anything).
				Return(&common.Status{ErrorCode: common.ErrorCode_UnexpectedError}, nil)

			err := c.ReleaseCollection(ctx, testCollectionName)
			s.Error(err)
		})
	})

	s.Run("service_not_ready", func() {
		c := &GrpcClient{}
		err := c.ReleaseCollection(ctx, testCollectionName)
		s.ErrorIs(err, ErrClientNotReady)
	})
}

func (s *CollectionSuite) TestHasCollection() {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	c := s.client

	s.Run("normal_run", func() {
		defer s.resetMock()

		s.setupDescribeCollection(testCollectionName, nil)
		s.mock.EXPECT().DescribeCollection(mock.Anything, mock.Anything).Run(func(_ context.Context, req *server.DescribeCollectionRequest) {
			s.Equal(testCollectionName, req.GetCollectionName())
		}).Return(&server.DescribeCollectionResponse{Status: &common.Status{ErrorCode: common.ErrorCode_Success}}, nil)

		result, err := c.HasCollection(ctx, testCollectionName)
		s.NoError(err)
		s.True(result)
	})

	s.Run("coll_not_exists", func() {
		defer s.resetMock()

		s.setupDescribeCollectionError(common.ErrorCode_CollectionNotExists, nil)

		result, err := c.HasCollection(ctx, testCollectionName)
		s.NoError(err)
		s.False(result)
	})

	s.Run("return_unexpected_oldreason", func() {
		defer s.resetMock()

		s.mock.EXPECT().DescribeCollection(mock.Anything, mock.Anything).
			Return(&server.DescribeCollectionResponse{
				Status: &common.Status{ErrorCode: common.ErrorCode_UnexpectedError, Reason: "can't find collection"},
			}, nil)

		result, err := c.HasCollection(ctx, testCollectionName)
		s.NoError(err)
		s.False(result)
	})

	s.Run("return_unexpected_other", func() {
		defer s.resetMock()

		s.mock.EXPECT().DescribeCollection(mock.Anything, mock.Anything).
			Return(&server.DescribeCollectionResponse{
				Status: &common.Status{ErrorCode: common.ErrorCode_UnexpectedError, Reason: "server not ready"},
			}, nil)

		result, err := c.HasCollection(ctx, testCollectionName)
		s.Error(err)
		s.False(result)
	})

	s.Run("return_error", func() {
		defer s.resetMock()

		s.setupDescribeCollectionError(common.ErrorCode_UnexpectedError, errors.New("mocked"))
		_, err := c.HasCollection(ctx, testCollectionName)
		s.Error(err)
	})

	s.Run("service_not_ready", func() {
		c := &GrpcClient{}
		_, err := c.HasCollection(ctx, testCollectionName)
		s.ErrorIs(err, ErrClientNotReady)
	})
}

func (s *CollectionSuite) TestDescribeCollection() {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	c := s.client
	sch := entity.NewSchema().WithName(testCollectionName).
		WithField(entity.NewField().WithName("ID").WithDataType(entity.FieldTypeInt64).WithIsPrimaryKey(true)).
		WithField(entity.NewField().WithName("vector").WithDataType(entity.FieldTypeFloatVector).WithDim(testVectorDim))

	s.Run("normal_run", func() {
		defer s.resetMock()

		s.setupDescribeCollection(testCollectionName, sch)

		coll, err := c.DescribeCollection(ctx, testCollectionName)
		s.NoError(err)
		s.Equal(testCollectionName, coll.Schema.CollectionName)
	})

	s.Run("describe_fail", func() {
		s.Run("return_err", func() {
			defer s.resetMock()

			s.setupDescribeCollectionError(common.ErrorCode_UnexpectedError, errors.New("mocked"))

			_, err := c.DescribeCollection(ctx, testCollectionName)
			s.Error(err)
		})

		s.Run("return_err_status", func() {
			defer s.resetMock()

			s.setupDescribeCollectionError(common.ErrorCode_UnexpectedError, nil)

			_, err := c.DescribeCollection(ctx, testCollectionName)
			s.Error(err)
		})
	})

	s.Run("service_not_ready", func() {
		c := &GrpcClient{}
		_, err := c.DescribeCollection(ctx, testCollectionName)
		s.ErrorIs(err, ErrClientNotReady)
	})
}

func (s *CollectionSuite) TestGetReplicas() {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	c := s.client
	replicaID := rand.Int63()
	nodeIds := []int64{1, 2, 3, 4}

	s.Run("normal_run", func() {
		defer s.resetMock()

		s.setupShowCollection(testCollectionName)
		s.mock.EXPECT().GetReplicas(mock.Anything, mock.Anything).Run(func(_ context.Context, req *server.GetReplicasRequest) {
			s.Equal(testCollectionID, req.GetCollectionID())
			s.True(req.GetWithShardNodes())
		}).Return(&server.GetReplicasResponse{
			Status: &common.Status{},
			Replicas: []*server.ReplicaInfo{{
				ReplicaID: replicaID,
				ShardReplicas: []*server.ShardReplica{
					{
						LeaderID:      1,
						DmChannelName: "DML_channel_v1",
					},
					{
						LeaderID:   2,
						LeaderAddr: "DML_channel_v2",
					},
				},
				NodeIds: nodeIds,
			}},
		}, nil)

		groups, err := c.GetReplicas(ctx, testCollectionName)
		s.NoError(err)
		s.NotNil(groups)
		s.Equal(1, len(groups))

		s.Equal(replicaID, groups[0].ReplicaID)
		s.Equal(nodeIds, groups[0].NodeIDs)
		s.Equal(2, len(groups[0].ShardReplicas))
	})

	s.Run("collection_not_exists", func() {
		defer s.resetMock()

		s.setupShowCollectionError(common.ErrorCode_CollectionNotExists, nil)

		_, err := c.GetReplicas(ctx, testCollectionName)
		s.Error(err)
	})

	s.Run("get_replica_fail", func() {
		s.Run("return_err", func() {
			defer s.resetMock()

			s.setupShowCollection(testCollectionName)
			s.mock.EXPECT().GetReplicas(mock.Anything, mock.Anything).
				Return(nil, errors.New("mocked"))

			_, err := c.GetReplicas(ctx, testCollectionName)
			s.Error(err)
		})

		s.Run("return_fail_status", func() {
			defer s.resetMock()

			s.setupShowCollection(testCollectionName)
			s.mock.EXPECT().GetReplicas(mock.Anything, mock.Anything).
				Return(&server.GetReplicasResponse{
					Status: &common.Status{ErrorCode: common.ErrorCode_UnexpectedError}}, nil)

			_, err := c.GetReplicas(ctx, testCollectionName)
			s.Error(err)
		})
	})

	s.Run("service_not_ready", func() {
		c := &GrpcClient{}
		_, err := c.GetReplicas(ctx, testCollectionName)
		s.ErrorIs(err, ErrClientNotReady)
	})
}

func TestCollectionSuite(t *testing.T) {
	suite.Run(t, new(CollectionSuite))
}
