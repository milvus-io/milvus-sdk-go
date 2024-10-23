package client

import (
	"context"
	"fmt"
	"math/rand"
	"testing"

	"github.com/cockroachdb/errors"

	"github.com/golang/protobuf/proto"
	"github.com/milvus-io/milvus-proto/go-api/v2/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v2/milvuspb"
	"github.com/milvus-io/milvus-proto/go-api/v2/schemapb"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/suite"
)

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
				Return(&milvuspb.ShowCollectionsResponse{
					Status:              &commonpb.Status{},
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
			Run(func(ctx context.Context, req *milvuspb.CreateCollectionRequest) {
				s.Equal(testCollectionName, req.GetCollectionName())
				sschema := &schemapb.CollectionSchema{}
				s.Require().NoError(proto.Unmarshal(req.GetSchema(), sschema))
				s.Require().Equal(len(ds.Fields), len(sschema.Fields))
				for idx, fieldSchema := range ds.Fields {
					s.Equal(fieldSchema.Name, sschema.GetFields()[idx].GetName())
					s.Equal(fieldSchema.PrimaryKey, sschema.GetFields()[idx].GetIsPrimaryKey())
					s.Equal(fieldSchema.AutoID, sschema.GetFields()[idx].GetAutoID())
					s.EqualValues(fieldSchema.DataType, sschema.GetFields()[idx].GetDataType())
				}
				s.Equal(shardsNum, req.GetShardsNum())
				s.Equal(commonpb.ConsistencyLevel_Bounded, req.GetConsistencyLevel())
			}).
			Return(&commonpb.Status{ErrorCode: commonpb.ErrorCode_Success}, nil)
		s.mock.EXPECT().HasCollection(mock.Anything, &milvuspb.HasCollectionRequest{CollectionName: testCollectionName}).Return(&milvuspb.BoolResponse{Status: &commonpb.Status{}, Value: false}, nil)

		err := c.CreateCollection(ctx, ds, shardsNum, WithCreateCollectionMsgBase(&commonpb.MsgBase{}))
		s.NoError(err)
	})

	s.Run("create_with_consistency_level", func() {
		ds := defaultSchema()
		shardsNum := int32(1)
		defer s.resetMock()
		s.mock.EXPECT().CreateCollection(mock.Anything, mock.AnythingOfType("*milvuspb.CreateCollectionRequest")).
			Run(func(ctx context.Context, req *milvuspb.CreateCollectionRequest) {
				s.Equal(testCollectionName, req.GetCollectionName())
				sschema := &schemapb.CollectionSchema{}
				s.Require().NoError(proto.Unmarshal(req.GetSchema(), sschema))
				s.Require().Equal(len(ds.Fields), len(sschema.Fields))
				for idx, fieldSchema := range ds.Fields {
					s.Equal(fieldSchema.Name, sschema.GetFields()[idx].GetName())
					s.Equal(fieldSchema.PrimaryKey, sschema.GetFields()[idx].GetIsPrimaryKey())
					s.Equal(fieldSchema.AutoID, sschema.GetFields()[idx].GetAutoID())
					s.EqualValues(fieldSchema.DataType, sschema.GetFields()[idx].GetDataType())
				}
				s.Equal(shardsNum, req.GetShardsNum())
				s.Equal(commonpb.ConsistencyLevel_Eventually, req.GetConsistencyLevel())
			}).
			Return(&commonpb.Status{ErrorCode: commonpb.ErrorCode_Success}, nil)
		s.mock.EXPECT().HasCollection(mock.Anything, &milvuspb.HasCollectionRequest{CollectionName: testCollectionName}).Return(&milvuspb.BoolResponse{Status: &commonpb.Status{}, Value: false}, nil)

		err := c.CreateCollection(ctx, ds, shardsNum, WithConsistencyLevel(entity.ClEventually))
		s.NoError(err)
	})

	s.Run("create_with_partition_key_isolation", func() {
		ds := defaultSchema()
		partitionKeyFieldName := "partitionKeyField"
		partitionKeyField := entity.NewField().WithName(partitionKeyFieldName).WithDataType(entity.FieldTypeInt64).WithIsPartitionKey(true)
		ds.WithField(partitionKeyField)

		shardsNum := int32(1)
		expectedIsoStr := "true"
		IsoKeystr := "partitionkey.isolation"
		defer s.resetMock()
		s.mock.EXPECT().CreateCollection(mock.Anything, mock.AnythingOfType("*milvuspb.CreateCollectionRequest")).
			Run(func(ctx context.Context, req *milvuspb.CreateCollectionRequest) {
				s.Equal(testCollectionName, req.GetCollectionName())
				sschema := &schemapb.CollectionSchema{}
				s.Require().NoError(proto.Unmarshal(req.GetSchema(), sschema))
				s.Require().Equal(len(ds.Fields), len(sschema.Fields))
				for idx, fieldSchema := range ds.Fields {
					s.Equal(fieldSchema.Name, sschema.GetFields()[idx].GetName())
					s.Equal(fieldSchema.PrimaryKey, sschema.GetFields()[idx].GetIsPrimaryKey())
					s.Equal(fieldSchema.AutoID, sschema.GetFields()[idx].GetAutoID())
					s.EqualValues(fieldSchema.DataType, sschema.GetFields()[idx].GetDataType())
				}
				s.Equal(shardsNum, req.GetShardsNum())
				s.Equal(commonpb.ConsistencyLevel_Eventually, req.GetConsistencyLevel())
				propsMp := make(map[string]string)
				for _, prop := range req.GetProperties() {
					propsMp[prop.Key] = prop.Value
				}
				s.Equal(expectedIsoStr, propsMp[IsoKeystr])
			}).
			Return(&commonpb.Status{ErrorCode: commonpb.ErrorCode_Success}, nil)
		s.mock.EXPECT().HasCollection(mock.Anything, &milvuspb.HasCollectionRequest{CollectionName: testCollectionName}).Return(&milvuspb.BoolResponse{Status: &commonpb.Status{}, Value: false}, nil)

		err := c.CreateCollection(ctx, ds, shardsNum, WithConsistencyLevel(entity.ClEventually), WithCollectionProperty(IsoKeystr, expectedIsoStr))
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
				name: "multiple primary key",
				schema: entity.NewSchema().WithName(testCollectionName).
					WithField(entity.NewField().WithName("int64").WithDataType(entity.FieldTypeInt64).WithIsPrimaryKey(true)).
					WithField(entity.NewField().WithName("int64_2").WithDataType(entity.FieldTypeInt64).WithIsPrimaryKey(true)).
					WithField(entity.NewField().WithName("vector").WithDataType(entity.FieldTypeFloatVector).WithDim(128)),
			},
			{
				name: "multiple auto id",
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
			s.mock.EXPECT().HasCollection(mock.Anything, &milvuspb.HasCollectionRequest{CollectionName: testCollectionName}).Return(&milvuspb.BoolResponse{Status: &commonpb.Status{}, Value: false}, nil)
			s.mock.EXPECT().CreateCollection(mock.Anything, mock.AnythingOfType("*milvuspb.CreateCollectionRequest")).
				Return(nil, errors.New("mocked grpc error"))

			err := c.CreateCollection(ctx, defaultSchema(), 1)
			s.Error(err)
		})

		s.Run("create_collection_fail", func() {
			defer s.resetMock()
			s.mock.EXPECT().HasCollection(mock.Anything, &milvuspb.HasCollectionRequest{CollectionName: testCollectionName}).Return(&milvuspb.BoolResponse{Status: &commonpb.Status{}, Value: false}, nil)
			s.mock.EXPECT().CreateCollection(mock.Anything, mock.AnythingOfType("*milvuspb.CreateCollectionRequest")).
				Return(&commonpb.Status{ErrorCode: commonpb.ErrorCode_UnexpectedError}, nil)

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
				c.config.addFlags(tc.flag)
				defer c.config.resetFlags(tc.flag)

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
			Run(func(ctx context.Context, req *milvuspb.CreateCollectionRequest) {
				s.Equal(testCollectionName, req.GetCollectionName())
				sschema := &schemapb.CollectionSchema{}
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
			Return(&commonpb.Status{ErrorCode: commonpb.ErrorCode_Success}, nil)
		s.mock.EXPECT().HasCollection(mock.Anything, &milvuspb.HasCollectionRequest{CollectionName: testCollectionName}).Call.Return(func(_ context.Context, _ *milvuspb.HasCollectionRequest) *milvuspb.BoolResponse {
			return &milvuspb.BoolResponse{Status: &commonpb.Status{}, Value: created}
		}, nil)
		s.mock.EXPECT().CreateIndex(mock.Anything, mock.Anything).Return(&commonpb.Status{ErrorCode: commonpb.ErrorCode_Success}, nil)
		s.mock.EXPECT().Flush(mock.Anything, mock.Anything).Return(&milvuspb.FlushResponse{
			Status:     &commonpb.Status{ErrorCode: commonpb.ErrorCode_Success},
			CollSegIDs: map[string]*schemapb.LongArray{},
		}, nil)
		s.mock.EXPECT().DescribeIndex(mock.Anything, mock.Anything).Return(&milvuspb.DescribeIndexResponse{
			Status: &commonpb.Status{ErrorCode: commonpb.ErrorCode_Success},
			IndexDescriptions: []*milvuspb.IndexDescription{
				{FieldName: "vector", State: commonpb.IndexState_Finished},
			},
		}, nil)
		s.mock.EXPECT().LoadCollection(mock.Anything, mock.Anything).Return(&commonpb.Status{ErrorCode: commonpb.ErrorCode_Success}, nil)
		s.mock.EXPECT().GetLoadingProgress(mock.Anything, mock.Anything).Return(&milvuspb.GetLoadingProgressResponse{
			Status:   &commonpb.Status{ErrorCode: commonpb.ErrorCode_Success},
			Progress: 100,
		}, nil)
		s.mock.EXPECT().DescribeCollection(mock.Anything, mock.Anything).Return(&milvuspb.DescribeCollectionResponse{
			Status: &commonpb.Status{ErrorCode: commonpb.ErrorCode_Success},
			Schema: &schemapb.CollectionSchema{
				Fields: []*schemapb.FieldSchema{
					{Name: "id", DataType: schemapb.DataType_VarChar},
					{Name: "vector", DataType: schemapb.DataType_FloatVector},
				},
			},
		}, nil)

		err := c.NewCollection(ctx, testCollectionName, testVectorDim)
		s.NoError(err)
	})

	s.Run("with_custom_set", func() {
		defer s.resetMock()
		created := false
		s.mock.EXPECT().CreateCollection(mock.Anything, mock.AnythingOfType("*milvuspb.CreateCollectionRequest")).
			Run(func(ctx context.Context, req *milvuspb.CreateCollectionRequest) {
				s.Equal(testCollectionName, req.GetCollectionName())
				sschema := &schemapb.CollectionSchema{}
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
			Return(&commonpb.Status{ErrorCode: commonpb.ErrorCode_Success}, nil)
		s.mock.EXPECT().HasCollection(mock.Anything, &milvuspb.HasCollectionRequest{CollectionName: testCollectionName}).Call.Return(func(_ context.Context, _ *milvuspb.HasCollectionRequest) *milvuspb.BoolResponse {
			return &milvuspb.BoolResponse{Status: &commonpb.Status{}, Value: created}
		}, nil)
		s.mock.EXPECT().CreateIndex(mock.Anything, mock.Anything).Return(&commonpb.Status{ErrorCode: commonpb.ErrorCode_Success}, nil)
		s.mock.EXPECT().Flush(mock.Anything, mock.Anything).Return(&milvuspb.FlushResponse{
			Status:     &commonpb.Status{ErrorCode: commonpb.ErrorCode_Success},
			CollSegIDs: map[string]*schemapb.LongArray{},
		}, nil)
		s.mock.EXPECT().DescribeIndex(mock.Anything, mock.Anything).Return(&milvuspb.DescribeIndexResponse{
			Status: &commonpb.Status{ErrorCode: commonpb.ErrorCode_Success},
			IndexDescriptions: []*milvuspb.IndexDescription{
				{FieldName: "embedding", State: commonpb.IndexState_Finished},
			},
		}, nil)
		s.mock.EXPECT().LoadCollection(mock.Anything, mock.Anything).Return(&commonpb.Status{ErrorCode: commonpb.ErrorCode_Success}, nil)
		s.mock.EXPECT().GetLoadingProgress(mock.Anything, mock.Anything).Return(&milvuspb.GetLoadingProgressResponse{
			Status:   &commonpb.Status{ErrorCode: commonpb.ErrorCode_Success},
			Progress: 100,
		}, nil)
		s.mock.EXPECT().DescribeCollection(mock.Anything, mock.Anything).Return(&milvuspb.DescribeCollectionResponse{
			Status: &commonpb.Status{ErrorCode: commonpb.ErrorCode_Success},
			Schema: &schemapb.CollectionSchema{
				Fields: []*schemapb.FieldSchema{
					{Name: "my_pk", DataType: schemapb.DataType_VarChar},
					{Name: "embedding", DataType: schemapb.DataType_FloatVector},
				},
			},
		}, nil)

		err := c.NewCollection(ctx, testCollectionName, testVectorDim, WithPKFieldName("my_pk"), WithPKFieldType(entity.FieldTypeVarChar), WithVectorFieldName("embedding"), WithConsistencyLevel(entity.ClEventually))
		s.NoError(err)
	})
}

func (s *CollectionSuite) TestRenameCollection() {
	c := s.client
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	s.Run("normal_run", func() {
		defer s.resetMock()

		newCollName := fmt.Sprintf("new_%s", randStr(6))

		s.mock.EXPECT().RenameCollection(mock.Anything, &milvuspb.RenameCollectionRequest{OldName: testCollectionName, NewName: newCollName}).Return(&commonpb.Status{}, nil)

		err := c.RenameCollection(ctx, testCollectionName, newCollName)
		s.NoError(err)
	})

	s.Run("rename_failed", func() {
		defer s.resetMock()

		newCollName := fmt.Sprintf("new_%s", randStr(6))

		s.mock.EXPECT().RenameCollection(mock.Anything, &milvuspb.RenameCollectionRequest{OldName: testCollectionName, NewName: newCollName}).Return(&commonpb.Status{ErrorCode: commonpb.ErrorCode_UnexpectedError, Reason: "mocked failure"}, nil)

		err := c.RenameCollection(ctx, testCollectionName, newCollName)
		s.Error(err)
	})

	s.Run("rename_error", func() {
		defer s.resetMock()

		newCollName := fmt.Sprintf("new_%s", randStr(6))

		s.mock.EXPECT().RenameCollection(mock.Anything, &milvuspb.RenameCollectionRequest{OldName: testCollectionName, NewName: newCollName}).Return(nil, errors.New("mocked error"))

		err := c.RenameCollection(ctx, testCollectionName, newCollName)
		s.Error(err)
	})
}

func (s *CollectionSuite) TestAlterCollection() {
	c := s.client
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	s.Run("normal_run", func() {
		defer s.resetMock()

		s.mock.EXPECT().AlterCollection(mock.Anything, mock.AnythingOfType("*milvuspb.AlterCollectionRequest")).
			Return(&commonpb.Status{}, nil)

		err := c.AlterCollection(ctx, testCollectionName, entity.CollectionTTL(100000))
		s.NoError(err)

		err = c.AlterCollection(ctx, testCollectionName, entity.CollectionPartitionKeyIsolation(true))
		s.NoError(err)
	})

	s.Run("no_attributes", func() {
		defer s.resetMock()

		err := c.AlterCollection(ctx, testCollectionName)
		s.Error(err)
	})

	s.Run("request_fails", func() {
		defer s.resetMock()
		s.mock.EXPECT().AlterCollection(mock.Anything, mock.AnythingOfType("*milvuspb.AlterCollectionRequest")).
			Return(nil, errors.New("mocked"))

		err := c.AlterCollection(ctx, testCollectionName, entity.CollectionTTL(100000))
		s.Error(err)
	})

	s.Run("server_return_error", func() {
		defer s.resetMock()

		s.mock.EXPECT().AlterCollection(mock.Anything, mock.AnythingOfType("*milvuspb.AlterCollectionRequest")).
			Return(&commonpb.Status{ErrorCode: commonpb.ErrorCode_UnexpectedError}, nil)

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

		s.mock.EXPECT().LoadCollection(mock.Anything, mock.Anything).Return(&commonpb.Status{ErrorCode: commonpb.ErrorCode_Success}, nil)

		err := c.LoadCollection(ctx, testCollectionName, true, WithLoadCollectionMsgBase(&commonpb.MsgBase{}))
		s.NoError(err)
	})

	s.Run("normal_run_sync", func() {
		defer s.resetMock()

		s.mock.EXPECT().LoadCollection(mock.Anything, mock.Anything).Return(&commonpb.Status{ErrorCode: commonpb.ErrorCode_Success}, nil)
		s.mock.EXPECT().GetLoadingProgress(mock.Anything, mock.Anything).
			Return(&milvuspb.GetLoadingProgressResponse{
				Status:   &commonpb.Status{ErrorCode: commonpb.ErrorCode_Success},
				Progress: 100,
			}, nil)

		err := c.LoadCollection(ctx, testCollectionName, true)
		s.NoError(err)
	})

	s.Run("load_default_replica", func() {
		defer s.resetMock()

		s.mock.EXPECT().LoadCollection(mock.Anything, mock.Anything).Run(func(_ context.Context, req *milvuspb.LoadCollectionRequest) {
			s.Equal(int32(0), req.GetReplicaNumber())
			s.Equal(testCollectionName, req.GetCollectionName())
		}).Return(&commonpb.Status{ErrorCode: commonpb.ErrorCode_Success}, nil)

		err := c.LoadCollection(ctx, testCollectionName, true)
		s.NoError(err)
	})

	s.Run("load_multiple_replica", func() {
		defer s.resetMock()

		s.mock.EXPECT().LoadCollection(mock.Anything, mock.Anything).Run(func(_ context.Context, req *milvuspb.LoadCollectionRequest) {
			s.Equal(testMultiReplicaNumber, req.GetReplicaNumber())
			s.Equal(testCollectionName, req.GetCollectionName())
		}).Return(&commonpb.Status{ErrorCode: commonpb.ErrorCode_Success}, nil)

		err := c.LoadCollection(ctx, testCollectionName, true, WithReplicaNumber(testMultiReplicaNumber))
		s.NoError(err)
	})

	s.Run("load_collection_failure", func() {
		s.Run("failure_status", func() {
			defer s.resetMock()

			s.mock.EXPECT().LoadCollection(mock.Anything, mock.Anything).
				Return(&commonpb.Status{ErrorCode: commonpb.ErrorCode_UnexpectedError}, nil)

			err := c.LoadCollection(ctx, testCollectionName, true)
			s.Error(err)
		})

		s.Run("return_error", func() {
			s.mock.EXPECT().LoadCollection(mock.Anything, mock.Anything).
				Return(nil, errors.New("mock error"))

			err := c.LoadCollection(ctx, testCollectionName, true)
			s.Error(err)
		})
	})

	s.Run("get_loading_progress_failure", func() {
		defer s.resetMock()

		s.mock.EXPECT().LoadCollection(mock.Anything, mock.Anything).Return(&commonpb.Status{ErrorCode: commonpb.ErrorCode_Success}, nil)
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
	c := s.client
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	collectionName := fmt.Sprintf("coll_%s", randStr(6))

	s.Run("normal_run", func() {
		defer s.resetMock()

		s.mock.EXPECT().DropCollection(mock.Anything, mock.AnythingOfType("*milvuspb.DropCollectionRequest")).RunAndReturn(func(ctx context.Context, dcr *milvuspb.DropCollectionRequest) (*commonpb.Status, error) {
			s.Equal(collectionName, dcr.GetCollectionName())
			return s.getSuccessStatus(), nil
		}).Once()

		err := c.DropCollection(ctx, collectionName)
		s.NoError(err)
	})

	s.Run("return_error", func() {
		defer s.resetMock()

		s.mock.EXPECT().DropCollection(mock.Anything, mock.AnythingOfType("*milvuspb.DropCollectionRequest")).RunAndReturn(func(ctx context.Context, dcr *milvuspb.DropCollectionRequest) (*commonpb.Status, error) {
			s.Equal(collectionName, dcr.GetCollectionName())
			return nil, errors.New("mock")
		}).Once()

		err := c.DropCollection(ctx, collectionName)
		s.Error(err)
	})
}

func (s *CollectionSuite) TestGetCollectionStatistics() {
	c := s.client
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	collectionName := fmt.Sprintf("coll_%s", randStr(6))
	stat := make(map[string]string)
	stat["row_count"] = fmt.Sprintf("%d", rand.Int31n(10000))

	s.Run("normal_case", func() {
		defer s.resetMock()

		s.mock.EXPECT().GetCollectionStatistics(mock.Anything, mock.AnythingOfType("*milvuspb.GetCollectionStatisticsRequest")).RunAndReturn(func(ctx context.Context, dcr *milvuspb.GetCollectionStatisticsRequest) (*milvuspb.GetCollectionStatisticsResponse, error) {
			s.Equal(collectionName, dcr.GetCollectionName())
			return &milvuspb.GetCollectionStatisticsResponse{
				Status: s.getSuccessStatus(),
				Stats:  entity.MapKvPairs(stat),
			}, nil
		}).Once()

		result, err := c.GetCollectionStatistics(ctx, collectionName)
		s.NoError(err)
		s.Len(result, len(stat))
		for k, v := range result {
			s.Equal(v, stat[k])
		}
	})

	s.Run("server_error", func() {
		defer s.resetMock()

		s.mock.EXPECT().GetCollectionStatistics(mock.Anything, mock.AnythingOfType("*milvuspb.GetCollectionStatisticsRequest")).RunAndReturn(func(ctx context.Context, dcr *milvuspb.GetCollectionStatisticsRequest) (*milvuspb.GetCollectionStatisticsResponse, error) {
			s.Equal(collectionName, dcr.GetCollectionName())
			return nil, errors.New("mock")
		}).Once()

		_, err := c.GetCollectionStatistics(ctx, collectionName)
		s.Error(err)
	})
}

func (s *CollectionSuite) TestReleaseCollection() {
	c := s.client
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	collectionName := fmt.Sprintf("coll_%s", randStr(6))

	s.Run("normal_run", func() {
		defer s.resetMock()

		s.mock.EXPECT().ReleaseCollection(mock.Anything, mock.AnythingOfType("*milvuspb.ReleaseCollectionRequest")).RunAndReturn(func(ctx context.Context, dcr *milvuspb.ReleaseCollectionRequest) (*commonpb.Status, error) {
			s.Equal(collectionName, dcr.GetCollectionName())
			return s.getSuccessStatus(), nil
		}).Once()

		err := c.ReleaseCollection(ctx, collectionName)
		s.NoError(err)
	})

	s.Run("return_error", func() {
		defer s.resetMock()

		s.mock.EXPECT().ReleaseCollection(mock.Anything, mock.AnythingOfType("*milvuspb.ReleaseCollectionRequest")).RunAndReturn(func(ctx context.Context, dcr *milvuspb.ReleaseCollectionRequest) (*commonpb.Status, error) {
			s.Equal(collectionName, dcr.GetCollectionName())
			return nil, errors.New("mock")
		}).Once()

		err := c.ReleaseCollection(ctx, collectionName)
		s.Error(err)
	})
}

func (s *CollectionSuite) TestGetLoadingProgress() {
	c := s.client
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	collectionName := fmt.Sprintf("coll_%s", randStr(6))
	partName := fmt.Sprintf("part_%s", randStr(6))
	loadProgress := rand.Int63n(100)

	s.Run("normal_run", func() {
		defer s.resetMock()

		s.mock.EXPECT().GetLoadingProgress(mock.Anything, mock.AnythingOfType("*milvuspb.GetLoadingProgressRequest")).RunAndReturn(func(ctx context.Context, glpr *milvuspb.GetLoadingProgressRequest) (*milvuspb.GetLoadingProgressResponse, error) {
			s.Equal(collectionName, glpr.GetCollectionName())
			return &milvuspb.GetLoadingProgressResponse{
				Status:   s.getSuccessStatus(),
				Progress: loadProgress,
			}, nil
		}).Once()

		percent, err := c.GetLoadingProgress(ctx, collectionName, []string{partName})
		s.NoError(err)
		s.Equal(loadProgress, percent)
	})

	s.Run("return_error", func() {
		defer s.resetMock()

		s.mock.EXPECT().GetLoadingProgress(mock.Anything, mock.AnythingOfType("*milvuspb.GetLoadingProgressRequest")).RunAndReturn(func(ctx context.Context, glpr *milvuspb.GetLoadingProgressRequest) (*milvuspb.GetLoadingProgressResponse, error) {
			s.Equal(collectionName, glpr.GetCollectionName())
			return nil, errors.New("mock")
		}).Once()

		_, err := c.GetLoadingProgress(ctx, collectionName, nil)
		s.Error(err)
	})
}

func (s *CollectionSuite) TestGetLoadState() {
	c := s.client
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	collectionName := fmt.Sprintf("coll_%s", randStr(6))
	partName := fmt.Sprintf("part_%s", randStr(6))
	loadState := commonpb.LoadState(rand.Int31n(4))

	s.Run("normal_run", func() {
		defer s.resetMock()

		s.mock.EXPECT().GetLoadState(mock.Anything, mock.AnythingOfType("*milvuspb.GetLoadStateRequest")).RunAndReturn(func(ctx context.Context, glsr *milvuspb.GetLoadStateRequest) (*milvuspb.GetLoadStateResponse, error) {
			s.Equal(collectionName, glsr.GetCollectionName())
			s.ElementsMatch([]string{partName}, glsr.GetPartitionNames())
			return &milvuspb.GetLoadStateResponse{
				Status: s.getSuccessStatus(),
				State:  loadState,
			}, nil
		}).Once()

		state, err := c.GetLoadState(ctx, collectionName, []string{partName})
		s.NoError(err)
		s.EqualValues(loadState, state)
	})

	s.Run("return_error", func() {
		defer s.resetMock()

		s.mock.EXPECT().GetLoadState(mock.Anything, mock.AnythingOfType("*milvuspb.GetLoadStateRequest")).RunAndReturn(func(ctx context.Context, glsr *milvuspb.GetLoadStateRequest) (*milvuspb.GetLoadStateResponse, error) {
			s.Equal(collectionName, glsr.GetCollectionName())
			return nil, errors.New("mock")
		}).Once()

		_, err := c.GetLoadState(ctx, collectionName, nil)
		s.Error(err)
	})
}

func TestCollectionSuite(t *testing.T) {
	suite.Run(t, new(CollectionSuite))
}

// default HasCollection injection, returns true only when collection name is `testCollectionName`
var hasCollectionDefault = func(_ context.Context, raw proto.Message) (proto.Message, error) {
	req, ok := raw.(*milvuspb.HasCollectionRequest)
	resp := &milvuspb.BoolResponse{}
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

func TestGrpcClientHasCollection(t *testing.T) {
	ctx := context.Background()

	c := testClient(ctx, t)

	mockServer.SetInjection(MHasCollection, func(_ context.Context, raw proto.Message) (proto.Message, error) {
		req, ok := raw.(*milvuspb.HasCollectionRequest)
		resp := &milvuspb.BoolResponse{}
		if !ok {
			s, err := BadRequestStatus()
			assert.Fail(t, err.Error())
			resp.Status = s
			return resp, err
		}
		assert.Equal(t, req.CollectionName, testCollectionName)

		s, err := SuccessStatus()
		resp.Status, resp.Value = s, true
		return resp, err
	})

	has, err := c.HasCollection(ctx, testCollectionName)
	assert.Nil(t, err)
	assert.True(t, has)
}

func describeCollectionInjection(t *testing.T, collID int64, collName string, sch *entity.Schema) func(_ context.Context, raw proto.Message) (proto.Message, error) {
	return func(_ context.Context, raw proto.Message) (proto.Message, error) {
		req, ok := raw.(*milvuspb.DescribeCollectionRequest)
		resp := &milvuspb.DescribeCollectionResponse{}
		if !ok {
			s, err := BadRequestStatus()
			resp.Status = s
			return resp, err
		}

		assert.Equal(t, collName, req.GetCollectionName())

		sch := sch
		resp.Schema = sch.ProtoMessage()
		resp.CollectionID = collID

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

func TestGrpcClientGetReplicas(t *testing.T) {
	ctx := context.Background()
	c := testClient(ctx, t)

	replicaID := rand.Int63()
	nodeIds := []int64{1, 2, 3, 4}
	mockServer.SetInjection(MHasCollection, hasCollectionDefault)
	defer mockServer.DelInjection(MHasCollection)

	mockServer.SetInjection(MShowCollections, func(_ context.Context, raw proto.Message) (proto.Message, error) {
		s, err := SuccessStatus()
		resp := &milvuspb.ShowCollectionsResponse{
			Status:              s,
			CollectionIds:       []int64{testCollectionID},
			CollectionNames:     []string{testCollectionName},
			InMemoryPercentages: []int64{100},
		}
		return resp, err
	})
	defer mockServer.DelInjection(MShowCollections)

	mockServer.SetInjection(MGetReplicas, func(ctx context.Context, raw proto.Message) (proto.Message, error) {
		req, ok := raw.(*milvuspb.GetReplicasRequest)
		resp := &milvuspb.GetReplicasResponse{}
		if !ok {
			s, err := BadRequestStatus()
			resp.Status = s
			return resp, err
		}

		assert.Equal(t, testCollectionID, req.CollectionID)

		s, err := SuccessStatus()
		resp.Status = s
		resp.Replicas = []*milvuspb.ReplicaInfo{{
			ReplicaID: replicaID,
			ShardReplicas: []*milvuspb.ShardReplica{
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
		}}
		return resp, err
	})

	t.Run("get replicas normal", func(t *testing.T) {
		groups, err := c.GetReplicas(ctx, testCollectionName)
		assert.Nil(t, err)
		assert.NotNil(t, groups)
		assert.Equal(t, 1, len(groups))

		assert.Equal(t, replicaID, groups[0].ReplicaID)
		assert.Equal(t, nodeIds, groups[0].NodeIDs)
		assert.Equal(t, 2, len(groups[0].ShardReplicas))
	})

	t.Run("get replicas grpc error", func(t *testing.T) {
		mockServer.SetInjection(MGetReplicas, func(ctx context.Context, raw proto.Message) (proto.Message, error) {
			return &milvuspb.GetReplicasResponse{}, errors.New("mockServer.d grpc error")
		})
		_, err := c.GetReplicas(ctx, testCollectionName)
		assert.Error(t, err)
	})

	t.Run("get replicas server error", func(t *testing.T) {
		mockServer.SetInjection(MGetReplicas, func(ctx context.Context, raw proto.Message) (proto.Message, error) {
			return &milvuspb.GetReplicasResponse{
				Status: &commonpb.Status{
					ErrorCode: commonpb.ErrorCode_UnexpectedError,
					Reason:    "Service is not healthy",
				},
				Replicas: nil,
			}, nil
		})
		_, err := c.GetReplicas(ctx, testCollectionName)
		assert.Error(t, err)
	})

	mockServer.DelInjection(MGetReplicas)
}
