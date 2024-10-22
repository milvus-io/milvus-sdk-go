// Copyright (C) 2019-2021 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License.

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
	"github.com/milvus-io/milvus-proto/go-api/v2/schemapb"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"
)

func TestGrpcClientFlush(t *testing.T) {
	ctx := context.Background()

	c := testClient(ctx, t)

	t.Run("test async flush", func(t *testing.T) {
		assert.Nil(t, c.Flush(ctx, testCollectionName, true, WithFlushMsgBase(&commonpb.MsgBase{})))
	})

	t.Run("test sync flush", func(t *testing.T) {
		// 1~10 segments
		segCount := rand.Intn(10) + 1
		segments := make([]int64, 0, segCount)
		for i := 0; i < segCount; i++ {
			segments = append(segments, rand.Int63())
		}
		// 510ms ~ 2s
		flushTime := 510 + rand.Intn(1500)
		start := time.Now()
		flag := false
		mockServer.SetInjection(MFlush, func(_ context.Context, raw proto.Message) (proto.Message, error) {
			req, ok := raw.(*milvuspb.FlushRequest)
			resp := &milvuspb.FlushResponse{}
			if !ok {
				s, err := BadRequestStatus()
				resp.Status = s
				return resp, err
			}
			assert.ElementsMatch(t, []string{testCollectionName}, req.GetCollectionNames())

			resp.CollSegIDs = make(map[string]*schemapb.LongArray)
			resp.CollSegIDs[testCollectionName] = &schemapb.LongArray{
				Data: segments,
			}

			s, err := SuccessStatus()
			resp.Status = s
			return resp, err
		})

		mockServer.SetInjection(MGetFlushState, func(_ context.Context, raw proto.Message) (proto.Message, error) {
			req, ok := raw.(*milvuspb.GetFlushStateRequest)
			resp := &milvuspb.GetFlushStateResponse{}
			if !ok {
				s, err := BadRequestStatus()
				resp.Status = s
				return resp, err
			}
			assert.ElementsMatch(t, segments, req.GetSegmentIDs())
			resp.Flushed = false
			if time.Since(start) > time.Duration(flushTime)*time.Millisecond {
				resp.Flushed = true
				flag = true
			}

			s, err := SuccessStatus()
			resp.Status = s
			return resp, err
		})
		assert.Nil(t, c.Flush(ctx, testCollectionName, false))
		assert.True(t, flag)

		start = time.Now()
		flag = false
		quickCtx, cancel := context.WithTimeout(ctx, 10*time.Millisecond)
		defer cancel()
		assert.NotNil(t, c.Flush(quickCtx, testCollectionName, false))
	})
}

type SearchSuite struct {
	MockSuiteBase
	sch        *entity.Schema
	schDynamic *entity.Schema
}

func (s *SearchSuite) SetupSuite() {
	s.MockSuiteBase.SetupSuite()

	s.sch = entity.NewSchema().WithName(testCollectionName).
		WithField(entity.NewField().WithName("ID").WithDataType(entity.FieldTypeInt64).WithIsPrimaryKey(true)).
		WithField(entity.NewField().WithName("Attr").WithDataType(entity.FieldTypeInt64)).
		WithField(entity.NewField().WithName("vector").WithDataType(entity.FieldTypeFloatVector).WithDim(testVectorDim))
	s.schDynamic = entity.NewSchema().WithName(testCollectionName).WithDynamicFieldEnabled(true).
		WithField(entity.NewField().WithName("ID").WithDataType(entity.FieldTypeInt64).WithIsPrimaryKey(true)).
		WithField(entity.NewField().WithName("$meta").WithDataType(entity.FieldTypeJSON).WithIsDynamic(true)).
		WithField(entity.NewField().WithName("vector").WithDataType(entity.FieldTypeFloatVector).WithDim(testVectorDim))
}

func (s *SearchSuite) TestSearchFail() {
	c := s.client
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	partName := "part_1"
	vectors := generateFloatVector(10, testVectorDim)
	sp, err := entity.NewIndexFlatSearchParam()
	s.Require().NoError(err)
	s.resetMock()

	s.Run("service_not_ready", func() {
		_, err := (&GrpcClient{}).Search(ctx, testCollectionName, []string{}, partName, []string{"ID"}, []entity.Vector{entity.FloatVector(vectors[0])}, "vector",
			entity.L2, 5, sp, WithSearchQueryConsistencyLevel(entity.ClStrong))
		s.Error(err)
		s.ErrorIs(err, ErrClientNotReady)
	})

	s.Run("fail_describecollection_error", func() {
		defer s.resetMock()

		s.setupDescribeCollectionError(commonpb.ErrorCode_Success, errors.New("mock error"))

		_, err := c.Search(ctx, testCollectionName, []string{partName}, "", []string{"ID"}, []entity.Vector{entity.FloatVector(vectors[0])}, "vector",
			entity.L2, 5, sp, WithSearchQueryConsistencyLevel(entity.ClStrong))
		s.Error(err)
	})

	s.Run("fail_describecollection_errcode", func() {
		defer s.resetMock()

		s.setupDescribeCollectionError(commonpb.ErrorCode_UnexpectedError, nil)

		_, err := c.Search(ctx, testCollectionName, []string{partName}, "", []string{"ID"}, []entity.Vector{entity.FloatVector(vectors[0])}, "vector",
			entity.L2, 5, sp, WithSearchQueryConsistencyLevel(entity.ClStrong))
		s.Error(err)
	})

	s.Run("fail_guaranteed_non_custom_cl", func() {
		defer s.resetMock()

		s.setupDescribeCollection(testCollectionName, s.sch)

		_, err := c.Search(ctx, testCollectionName, []string{partName}, "", []string{"ID"}, []entity.Vector{entity.FloatVector(vectors[0])}, "vector",
			entity.L2, 5, sp, WithSearchQueryConsistencyLevel(entity.ClStrong), WithGuaranteeTimestamp(1000000))
		s.Error(err)
	})

	s.Run("fail_search_error", func() {
		defer s.resetMock()

		s.setupDescribeCollection(testCollectionName, s.sch)
		s.mock.EXPECT().Search(mock.Anything, mock.AnythingOfType("*milvuspb.SearchRequest")).
			Return(nil, errors.New("mock error"))

		_, err := c.Search(ctx, testCollectionName, []string{partName}, "", []string{"ID"}, []entity.Vector{entity.FloatVector(vectors[0])}, "vector",
			entity.L2, 5, sp, WithSearchQueryConsistencyLevel(entity.ClStrong))
		s.Error(err)
	})

	s.Run("fail_search_errcode", func() {
		defer s.resetMock()

		s.setupDescribeCollection(testCollectionName, s.sch)
		s.mock.EXPECT().Search(mock.Anything, mock.AnythingOfType("*milvuspb.SearchRequest")).
			Return(&milvuspb.SearchResults{Status: &commonpb.Status{ErrorCode: commonpb.ErrorCode_UnexpectedError}}, nil)

		_, err := c.Search(ctx, testCollectionName, []string{partName}, "", []string{"ID"}, []entity.Vector{entity.FloatVector(vectors[0])}, "vector",
			entity.L2, 5, sp, WithSearchQueryConsistencyLevel(entity.ClStrong))
		s.Error(err)
	})
}

func (s *SearchSuite) TestSearchSuccess() {
	c := s.client
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	partName := "part_1"
	vectors := generateFloatVector(10, testVectorDim)
	sp, err := entity.NewIndexFlatSearchParam()
	s.Require().NoError(err)
	s.resetMock()

	expr := "ID > 0"

	s.Run("non_dynamic_schema", func() {
		defer s.resetMock()
		s.setupDescribeCollection(testCollectionName, s.sch)
		s.mock.EXPECT().Search(mock.Anything, mock.AnythingOfType("*milvuspb.SearchRequest")).
			Run(func(_ context.Context, req *milvuspb.SearchRequest) {
				s.Equal(testCollectionName, req.GetCollectionName())
				s.Equal(expr, req.GetDsl())
				s.Equal(commonpb.DslType_BoolExprV1, req.GetDslType())
				s.ElementsMatch([]string{"ID"}, req.GetOutputFields())
				s.ElementsMatch([]string{partName}, req.GetPartitionNames())
			}).
			Return(&milvuspb.SearchResults{
				Status: getSuccessStatus(),
				Results: &schemapb.SearchResultData{
					NumQueries: 1,
					TopK:       10,
					FieldsData: []*schemapb.FieldData{
						s.getInt64FieldData("ID", []int64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}),
					},
					Ids: &schemapb.IDs{
						IdField: &schemapb.IDs_IntId{
							IntId: &schemapb.LongArray{
								Data: []int64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
							},
						},
					},
					Scores: make([]float32, 10),
					Topks:  []int64{10},
				},
			}, nil)

		r, err := c.Search(ctx, testCollectionName, []string{partName}, expr, []string{"ID"}, []entity.Vector{entity.FloatVector(vectors[0])},
			testVectorField, entity.L2, 10, sp, WithIgnoreGrowing(), WithForTuning(), WithSearchQueryConsistencyLevel(entity.ClCustomized), WithGuaranteeTimestamp(10000000000))
		s.NoError(err)
		s.Require().Equal(1, len(r))
		result := r[0]
		s.Require().NotNil(result.Fields.GetColumn("ID"))
	})

	s.Run("dynamic_schema", func() {
		defer s.resetMock()
		s.setupDescribeCollection(testCollectionName, s.schDynamic)
		s.mock.EXPECT().Search(mock.Anything, mock.AnythingOfType("*milvuspb.SearchRequest")).
			Run(func(_ context.Context, req *milvuspb.SearchRequest) {
				s.Equal(testCollectionName, req.GetCollectionName())
				s.Equal(expr, req.GetDsl())
				s.Equal(commonpb.DslType_BoolExprV1, req.GetDslType())
				s.ElementsMatch([]string{"A", "B"}, req.GetOutputFields())
				s.ElementsMatch([]string{partName}, req.GetPartitionNames())
			}).
			Return(&milvuspb.SearchResults{
				Status: getSuccessStatus(),
				Results: &schemapb.SearchResultData{
					NumQueries: 1,
					TopK:       2,
					FieldsData: []*schemapb.FieldData{
						s.getJSONBytesFieldData("", [][]byte{
							[]byte(`{"A": 123, "B": "456"}`),
							[]byte(`{"B": "abc", "A": 456}`),
						}, true),
					},
					Ids: &schemapb.IDs{
						IdField: &schemapb.IDs_IntId{
							IntId: &schemapb.LongArray{
								Data: []int64{1, 2},
							},
						},
					},
					Scores: make([]float32, 2),
					Topks:  []int64{2},
				},
			}, nil)

		r, err := c.Search(ctx, testCollectionName, []string{partName}, expr, []string{"A", "B"}, []entity.Vector{entity.FloatVector(vectors[0])},
			testVectorField, entity.L2, 2, sp, WithIgnoreGrowing(), WithForTuning(), WithSearchQueryConsistencyLevel(entity.ClBounded))
		s.NoError(err)
		s.Require().Equal(1, len(r))
		result := r[0]
		columnA := result.Fields.GetColumn("A")
		s.Require().NotNil(columnA)
		column, ok := columnA.(*entity.ColumnDynamic)
		s.Require().True(ok)
		v, err := column.GetAsInt64(0)
		s.NoError(err)
		s.Equal(int64(123), v)

		columnB := result.Fields.GetColumn("B")
		s.Require().NotNil(columnB)
		column, ok = columnB.(*entity.ColumnDynamic)
		s.Require().True(ok)
		str, err := column.GetAsString(1)
		s.NoError(err)
		s.Equal("abc", str)
	})

	s.Run("group_by", func() {
		defer s.resetMock()
		s.setupDescribeCollection(testCollectionName, s.sch)
		s.mock.EXPECT().Search(mock.Anything, mock.AnythingOfType("*milvuspb.SearchRequest")).
			Run(func(_ context.Context, req *milvuspb.SearchRequest) {
				s.Equal(testCollectionName, req.GetCollectionName())
				s.Equal(expr, req.GetDsl())
				s.Equal(commonpb.DslType_BoolExprV1, req.GetDslType())
				s.ElementsMatch([]string{"ID"}, req.GetOutputFields())
				s.ElementsMatch([]string{partName}, req.GetPartitionNames())
			}).
			Return(&milvuspb.SearchResults{
				Status: getSuccessStatus(),
				Results: &schemapb.SearchResultData{
					NumQueries: 1,
					TopK:       10,
					FieldsData: []*schemapb.FieldData{
						s.getInt64FieldData("ID", []int64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}),
					},
					Ids: &schemapb.IDs{
						IdField: &schemapb.IDs_IntId{
							IntId: &schemapb.LongArray{
								Data: []int64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
							},
						},
					},
					Scores:            make([]float32, 10),
					Topks:             []int64{10},
					GroupByFieldValue: s.getInt64FieldData("Attr", []int64{10, 10, 10, 10, 10, 10, 10, 10, 10, 10}),
				},
			}, nil)

		r, err := c.Search(ctx, testCollectionName, []string{partName}, expr, []string{"ID"}, []entity.Vector{entity.FloatVector(vectors[0])},
			testVectorField, entity.L2, 10, sp, WithIgnoreGrowing(), WithForTuning(), WithSearchQueryConsistencyLevel(entity.ClCustomized), WithGuaranteeTimestamp(10000000000), WithGroupByField("Attr"))
		s.NoError(err)
		s.Require().Equal(1, len(r))
		result := r[0]
		s.Require().NotNil(result.Fields.GetColumn("ID"))
	})
}

func TestSearch(t *testing.T) {
	suite.Run(t, new(SearchSuite))
}

type QuerySuite struct {
	MockSuiteBase
	sch        *entity.Schema
	schDynamic *entity.Schema
}

func (s *QuerySuite) SetupSuite() {
	s.MockSuiteBase.SetupSuite()

	s.sch = entity.NewSchema().WithName(testCollectionName).
		WithField(entity.NewField().WithName("ID").WithDataType(entity.FieldTypeInt64).WithIsPrimaryKey(true)).
		WithField(entity.NewField().WithName("vector").WithDataType(entity.FieldTypeFloatVector).WithDim(testVectorDim))
	s.schDynamic = entity.NewSchema().WithName(testCollectionName).WithDynamicFieldEnabled(true).
		WithField(entity.NewField().WithName("ID").WithDataType(entity.FieldTypeVarChar).WithIsPrimaryKey(true)).
		WithField(entity.NewField().WithName("$meta").WithDataType(entity.FieldTypeJSON).WithIsDynamic(true)).
		WithField(entity.NewField().WithName("vector").WithDataType(entity.FieldTypeFloatVector).WithDim(testVectorDim))
}

func (s *QuerySuite) GetFail() {
	c := s.client
	ctx, cancel := context.WithCancel((context.Background()))
	defer cancel()

	idCol := entity.NewColumnInt64("ID", []int64{1})
	s.Run("service_not_ready", func() {
		_, err := (&GrpcClient{}).Get(ctx, testCollectionName, idCol)
		s.Error(err)
		s.ErrorIs(err, ErrClientNotReady)
	})

	s.Run("ids_len_0", func() {
		_, err := c.Get(ctx, testCollectionName, entity.NewColumnInt64("ID", []int64{}), GetWithOutputFields("ID"))
		s.Error(err)
	})

	s.Run("describe_failed", func() {
		defer s.resetMock()
		s.setupDescribeCollectionError(commonpb.ErrorCode_Success, errors.New("mock error"))

		_, err := c.Get(ctx, testCollectionName, idCol)
		s.Error(err)
	})
}

func (s *QuerySuite) TestQueryByPksFail() {
	c := s.client
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	partName := "part_1"
	idCol := entity.NewColumnInt64("ID", []int64{1})
	s.Run("service_not_ready", func() {
		_, err := (&GrpcClient{}).QueryByPks(ctx, testCollectionName, []string{partName}, idCol, []string{"ID"})
		s.Error(err)
		s.ErrorIs(err, ErrClientNotReady)
	})

	s.Run("ids_len_0", func() {
		_, err := c.QueryByPks(ctx, testCollectionName, []string{partName}, entity.NewColumnInt64("ID", []int64{}), []string{"ID"})
		s.Error(err)
	})

	s.Run("query_failed", func() {
		defer s.resetMock()
		s.setupDescribeCollectionError(commonpb.ErrorCode_Success, errors.New("mock error"))

		_, err := c.QueryByPks(ctx, testCollectionName, []string{partName}, idCol, []string{"ID"})
		s.Error(err)
	})
}

func (s *QuerySuite) TestQueryFail() {
	c := s.client
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	partName := "part_1"
	s.resetMock()

	s.Run("service_not_ready", func() {
		_, err := (&GrpcClient{}).Query(ctx, testCollectionName, []string{partName}, "", []string{"ID"}, WithSearchQueryConsistencyLevel(entity.ClStrong))
		s.Error(err)
		s.ErrorIs(err, ErrClientNotReady)
	})

	s.Run("fail_describecollection_error", func() {
		defer s.resetMock()

		s.setupDescribeCollectionError(commonpb.ErrorCode_Success, errors.New("mock error"))

		_, err := c.Query(ctx, testCollectionName, []string{partName}, "", []string{"ID"}, WithSearchQueryConsistencyLevel(entity.ClStrong))
		s.Error(err)
	})

	s.Run("fail_describecollection_errcode", func() {
		defer s.resetMock()

		s.setupDescribeCollectionError(commonpb.ErrorCode_UnexpectedError, nil)

		_, err := c.Query(ctx, testCollectionName, []string{partName}, "", []string{"ID"}, WithSearchQueryConsistencyLevel(entity.ClStrong))
		s.Error(err)
	})

	s.Run("fail_guaranteed_non_custom_cl", func() {
		defer s.resetMock()

		s.setupDescribeCollection(testCollectionName, s.sch)

		_, err := c.Query(ctx, testCollectionName, []string{partName}, "", []string{"ID"}, WithSearchQueryConsistencyLevel(entity.ClStrong), WithGuaranteeTimestamp(1000000))
		s.Error(err)
	})

	s.Run("fail_search_error", func() {
		defer s.resetMock()

		s.setupDescribeCollection(testCollectionName, s.sch)
		s.mock.EXPECT().Query(mock.Anything, mock.AnythingOfType("*milvuspb.QueryRequest")).
			Return(nil, errors.New("mock error"))

		_, err := c.Query(ctx, testCollectionName, []string{partName}, "ID in {1}", []string{"ID"}, WithSearchQueryConsistencyLevel(entity.ClStrong))
		s.Error(err)
	})

	s.Run("fail_search_errcode", func() {
		defer s.resetMock()

		s.setupDescribeCollection(testCollectionName, s.sch)
		s.mock.EXPECT().Query(mock.Anything, mock.AnythingOfType("*milvuspb.QueryRequest")).
			Return(&milvuspb.QueryResults{Status: &commonpb.Status{ErrorCode: commonpb.ErrorCode_UnexpectedError}}, nil)

		_, err := c.Query(ctx, testCollectionName, []string{partName}, "ID in {1}", []string{"ID"}, WithSearchQueryConsistencyLevel(entity.ClStrong))
		s.Error(err)
	})

	s.Run("fail_response_type_error", func() {
		defer s.resetMock()

		s.setupDescribeCollection(testCollectionName, s.sch)
		s.mock.EXPECT().Query(mock.Anything, mock.AnythingOfType("*milvuspb.QueryRequest")).
			Return(&milvuspb.QueryResults{
				Status: &commonpb.Status{ErrorCode: commonpb.ErrorCode_Success},
				FieldsData: []*schemapb.FieldData{
					{
						FieldName: "ID",
						Type:      schemapb.DataType_String, // wrong data type here
						Field: &schemapb.FieldData_Scalars{
							Scalars: &schemapb.ScalarField{
								Data: &schemapb.ScalarField_LongData{
									LongData: &schemapb.LongArray{
										Data: []int64{1},
									},
								},
							},
						},
					},
				},
			}, nil)

		_, err := c.Query(ctx, testCollectionName, []string{partName}, "ID in {1}", []string{"ID"}, WithSearchQueryConsistencyLevel(entity.ClStrong))
		s.Error(err)
	})
}

func (s *QuerySuite) TestQuerySuccess() {
	c := s.client
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	partName := "part_1"
	s.resetMock()

	expr := "ID in {1}"

	s.Run("non_dynamic", func() {
		defer s.resetMock()

		s.setupDescribeCollection(testCollectionName, s.sch)
		s.mock.EXPECT().Query(mock.Anything, mock.AnythingOfType("*milvuspb.QueryRequest")).
			Run(func(_ context.Context, req *milvuspb.QueryRequest) {}).
			Return(&milvuspb.QueryResults{
				Status: getSuccessStatus(),
				FieldsData: []*schemapb.FieldData{
					s.getInt64FieldData("ID", []int64{1}),
					s.getFloatVectorFieldData("vector", 1, []float32{0.1}),
				},
			}, nil)

		rs, err := c.Query(ctx, testCollectionName, []string{partName}, expr, []string{"ID", "vector"}, WithSearchQueryConsistencyLevel(entity.ClStrong))
		s.NoError(err)
		s.Require().Equal(2, len(rs))
		colID, ok := rs.GetColumn("ID").(*entity.ColumnInt64)
		s.Require().True(ok)
		s.NotNil(colID)
		v, err := colID.Get(0)
		s.NoError(err)
		s.EqualValues(1, v)
		colVector, ok := rs.GetColumn("vector").(*entity.ColumnFloatVector)
		s.Require().True(ok)
		s.NotNil(colVector)
		v, err = colVector.Get(0)
		s.NoError(err)
		s.EqualValues([]float32{0.1}, v)
	})

	s.Run("dynamic_schema", func() {
		defer s.resetMock()

		s.setupDescribeCollection(testCollectionName, s.schDynamic)
		s.mock.EXPECT().Query(mock.Anything, mock.AnythingOfType("*milvuspb.QueryRequest")).
			Run(func(_ context.Context, req *milvuspb.QueryRequest) {}).
			Return(&milvuspb.QueryResults{
				Status: getSuccessStatus(),
				FieldsData: []*schemapb.FieldData{
					s.getVarcharFieldData("ID", []string{"1"}),
					s.getFloatVectorFieldData("vector", 1, []float32{0.1}),
					s.getJSONBytesFieldData("$meta", [][]byte{
						[]byte(`{"A": 123, "B": "456"}`),
						[]byte(`{"B": "abc", "A": 456}`),
					}, true),
				},
			}, nil)

		rs, err := c.Query(ctx, testCollectionName, []string{partName}, `id in {"1"}`, []string{"ID", "vector", "A"}, WithSearchQueryConsistencyLevel(entity.ClStrong))
		s.NoError(err)
		s.Require().Equal(3, len(rs))
		colID, ok := rs.GetColumn("ID").(*entity.ColumnVarChar)
		s.Require().True(ok)
		s.NotNil(colID)
		v, err := colID.Get(0)
		s.NoError(err)
		s.EqualValues("1", v)
		colVector, ok := rs.GetColumn("vector").(*entity.ColumnFloatVector)
		s.Require().True(ok)
		s.NotNil(colVector)
		v, err = colVector.Get(0)
		s.NoError(err)
		s.EqualValues([]float32{0.1}, v)

		columnA := rs.GetColumn("A").(*entity.ColumnDynamic)
		s.Require().True(ok)
		s.Require().NotNil(columnA)
		v, err = columnA.GetAsInt64(0)
		s.NoError(err)
		s.Equal(int64(123), v)
	})
}

func TestQuery(t *testing.T) {
	suite.Run(t, new(QuerySuite))
}

func TestGrpcCalcDistanceWithIDs(t *testing.T) {
	ctx := context.Background()
	t.Run("bad client calls CalcDistance", func(t *testing.T) {
		c := &GrpcClient{}
		r, err := c.CalcDistance(ctx, testCollectionName, []string{}, entity.L2, nil, nil)
		assert.Nil(t, r)
		assert.NotNil(t, err)
		assert.EqualValues(t, ErrClientNotReady, err)
	})

	c := testClient(ctx, t)
	mockServer.SetInjection(MDescribeCollection, func(_ context.Context, raw proto.Message) (proto.Message, error) {
		req, ok := raw.(*milvuspb.DescribeCollectionRequest)
		resp := &milvuspb.DescribeCollectionResponse{}
		if !ok {
			s, err := BadRequestStatus()
			resp.Status = s
			return resp, err
		}
		assert.Equal(t, testCollectionName, req.GetCollectionName())

		sch := defaultSchema()
		resp.Schema = sch.ProtoMessage()

		s, err := SuccessStatus()
		resp.Status = s
		return resp, err
	})
	t.Run("call with ctx done", func(t *testing.T) {
		ctxDone, cancel := context.WithCancel(context.Background())
		cancel()

		r, err := c.CalcDistance(ctxDone, testCollectionName, []string{}, entity.L2,
			entity.NewColumnInt64("int64", []int64{1}), entity.NewColumnInt64("int64", []int64{1}))
		assert.Nil(t, r)
		assert.NotNil(t, err)
	})

	t.Run("invalid ids call", func(t *testing.T) {
		r, err := c.CalcDistance(ctx, testCollectionName, []string{}, entity.L2,
			nil, nil)
		assert.Nil(t, r)
		assert.NotNil(t, err)

		r, err = c.CalcDistance(ctx, testCollectionName, []string{}, entity.L2,
			entity.NewColumnInt64("non-exists", []int64{1}), entity.NewColumnInt64("non-exists", []int64{1}))
		assert.Nil(t, r)
		assert.NotNil(t, err)

		r, err = c.CalcDistance(ctx, testCollectionName, []string{}, entity.L2,
			entity.NewColumnInt64("non-exists", []int64{1}), entity.NewColumnInt64("int64", []int64{1}))
		assert.Nil(t, r)
		assert.NotNil(t, err)
	})

	t.Run("valid calls", func(t *testing.T) {
		mockServer.SetInjection(MCalcDistance, func(_ context.Context, raw proto.Message) (proto.Message, error) {
			req, ok := raw.(*milvuspb.CalcDistanceRequest)
			resp := &milvuspb.CalcDistanceResults{}
			if !ok {
				s, err := BadRequestStatus()
				resp.Status = s
				return resp, err
			}
			idsLeft := req.GetOpLeft().GetIdArray()
			valuesLeft := req.GetOpLeft().GetDataArray()
			idsRight := req.GetOpRight().GetIdArray()
			valuesRight := req.GetOpRight().GetDataArray()
			assert.True(t, idsLeft != nil || valuesLeft != nil)
			assert.True(t, idsRight != nil || valuesRight != nil)

			if idsLeft != nil {
				assert.Equal(t, testCollectionName, idsLeft.CollectionName)
			}
			if idsRight != nil {
				assert.Equal(t, testCollectionName, idsRight.CollectionName)
			}

			// this injection returns float distance
			dl := 0
			if idsLeft != nil {
				dl = len(idsLeft.IdArray.GetIntId().GetData())
			}
			if valuesLeft != nil {
				dl = len(valuesLeft.GetFloatVector().GetData()) / int(valuesLeft.Dim)
			}
			dr := 0
			if idsRight != nil {
				dr = len(idsRight.IdArray.GetIntId().GetData())
			}
			if valuesRight != nil {
				dr = len(valuesRight.GetFloatVector().GetData()) / int(valuesRight.Dim)
			}

			resp.Array = &milvuspb.CalcDistanceResults_FloatDist{
				FloatDist: &schemapb.FloatArray{
					Data: make([]float32, dl*dr),
				},
			}

			s, err := SuccessStatus()
			resp.Status = s
			return resp, err
		})
		r, err := c.CalcDistance(ctx, testCollectionName, []string{}, entity.L2,
			entity.NewColumnInt64("vector", []int64{1}), entity.NewColumnInt64("vector", []int64{1}))
		assert.Nil(t, err)
		assert.NotNil(t, r)

		vectors := generateFloatVector(5, testVectorDim)
		r, err = c.CalcDistance(ctx, testCollectionName, []string{}, entity.L2,
			entity.NewColumnInt64("vector", []int64{1}), entity.NewColumnFloatVector("vector", testVectorDim, vectors))
		assert.Nil(t, err)
		assert.NotNil(t, r)

		r, err = c.CalcDistance(ctx, testCollectionName, []string{}, entity.L2,
			entity.NewColumnFloatVector("vector", testVectorDim, vectors), entity.NewColumnInt64("vector", []int64{1}))
		assert.Nil(t, err)
		assert.NotNil(t, r)

		// test IntDistance,
		mockServer.SetInjection(MCalcDistance, func(_ context.Context, raw proto.Message) (proto.Message, error) {
			req, ok := raw.(*milvuspb.CalcDistanceRequest)
			resp := &milvuspb.CalcDistanceResults{}
			if !ok {
				s, err := BadRequestStatus()
				resp.Status = s
				return resp, err
			}
			idsLeft := req.GetOpLeft().GetIdArray()
			valuesLeft := req.GetOpLeft().GetDataArray()
			idsRight := req.GetOpRight().GetIdArray()
			valuesRight := req.GetOpRight().GetDataArray()
			assert.True(t, idsLeft != nil || valuesLeft != nil)
			assert.True(t, idsRight != nil || valuesRight != nil)

			if idsLeft != nil {
				assert.Equal(t, testCollectionName, idsLeft.CollectionName)
			}
			if idsRight != nil {
				assert.Equal(t, testCollectionName, idsRight.CollectionName)
			}

			// this injection returns float distance
			dl := 0
			if idsLeft != nil {
				dl = len(idsLeft.IdArray.GetIntId().GetData())
			}
			if valuesLeft != nil {
				dl = len(valuesLeft.GetFloatVector().GetData()) / int(valuesLeft.Dim)
			}
			dr := 0
			if idsRight != nil {
				dr = len(idsRight.IdArray.GetIntId().GetData())
			}
			if valuesRight != nil {
				dr = len(valuesRight.GetFloatVector().GetData()) / int(valuesRight.Dim)
			}

			resp.Array = &milvuspb.CalcDistanceResults_IntDist{
				IntDist: &schemapb.IntArray{
					Data: make([]int32, dl*dr),
				},
			}

			s, err := SuccessStatus()
			resp.Status = s
			return resp, err
		})
		r, err = c.CalcDistance(ctx, testCollectionName, []string{}, entity.HAMMING,
			entity.NewColumnInt64("vector", []int64{1}), entity.NewColumnInt64("vector", []int64{1}))
		assert.Nil(t, err)
		assert.NotNil(t, r)

		// test str id
		mockServer.SetInjection(MDescribeCollection, func(_ context.Context, raw proto.Message) (proto.Message, error) {
			req, ok := raw.(*milvuspb.DescribeCollectionRequest)
			resp := &milvuspb.DescribeCollectionResponse{}
			if !ok {
				s, err := BadRequestStatus()
				resp.Status = s
				return resp, err
			}
			assert.Equal(t, testCollectionName, req.GetCollectionName())

			sch := defaultSchema()
			sch.Fields[0].DataType = entity.FieldTypeString
			sch.Fields[0].Name = "str"
			resp.Schema = sch.ProtoMessage()

			s, err := SuccessStatus()
			resp.Status = s
			return resp, err
		})
		mockServer.SetInjection(MCalcDistance, func(_ context.Context, raw proto.Message) (proto.Message, error) {
			req, ok := raw.(*milvuspb.CalcDistanceRequest)
			resp := &milvuspb.CalcDistanceResults{}
			if !ok {
				s, err := BadRequestStatus()
				resp.Status = s
				return resp, err
			}
			idsLeft := req.GetOpLeft().GetIdArray()
			idsRight := req.GetOpRight().GetIdArray()
			assert.NotNil(t, idsLeft)
			assert.NotNil(t, idsRight)

			assert.Equal(t, testCollectionName, idsLeft.CollectionName)
			assert.Equal(t, testCollectionName, idsRight.CollectionName)

			// only int ids supported for now TODO update string test cases
			assert.NotNil(t, idsLeft.IdArray.GetStrId())
			assert.NotNil(t, idsRight.IdArray.GetStrId())

			// this injection returns float distance
			dl := len(idsLeft.IdArray.GetStrId().GetData())

			resp.Array = &milvuspb.CalcDistanceResults_FloatDist{
				FloatDist: &schemapb.FloatArray{
					Data: make([]float32, dl),
				},
			}

			s, err := SuccessStatus()
			resp.Status = s
			return resp, err
		})
		r, err = c.CalcDistance(ctx, testCollectionName, []string{}, entity.L2,
			entity.NewColumnString("vector", []string{"1"}), entity.NewColumnString("vector", []string{"1"}))
		assert.Nil(t, err)
		assert.NotNil(t, r)
	})
}

func TestIsCollectionPrimaryKey(t *testing.T) {
	t.Run("nil cases", func(t *testing.T) {
		assert.False(t, isCollectionPrimaryKey(nil, nil))
		assert.False(t, isCollectionPrimaryKey(&entity.Collection{}, entity.NewColumnInt64("id", []int64{})))
	})

	t.Run("check cases", func(t *testing.T) {
		assert.False(t, isCollectionPrimaryKey(&entity.Collection{
			Schema: defaultSchema(),
		}, entity.NewColumnInt64("id", []int64{})))
		assert.False(t, isCollectionPrimaryKey(&entity.Collection{
			Schema: defaultSchema(),
		}, entity.NewColumnInt32("int64", []int32{})))
		assert.True(t, isCollectionPrimaryKey(&entity.Collection{
			Schema: defaultSchema(),
		}, entity.NewColumnInt64("int64", []int64{})))
	})
}

func TestEstRowSize(t *testing.T) {
	// a schema contains all supported vector
	sch := entity.NewSchema().WithName(testCollectionName).WithAutoID(false).
		WithField(entity.NewField().WithName(testPrimaryField).WithDataType(entity.FieldTypeInt64).WithIsPrimaryKey(true).WithIsAutoID(true)).
		WithField(entity.NewField().WithName("attr1").WithDataType(entity.FieldTypeInt8)).
		WithField(entity.NewField().WithName("attr2").WithDataType(entity.FieldTypeInt16)).
		WithField(entity.NewField().WithName("attr3").WithDataType(entity.FieldTypeInt32)).
		WithField(entity.NewField().WithName("attr4").WithDataType(entity.FieldTypeFloat)).
		WithField(entity.NewField().WithName("attr5").WithDataType(entity.FieldTypeDouble)).
		WithField(entity.NewField().WithName("attr6").WithDataType(entity.FieldTypeBool)).
		WithField(entity.NewField().WithName("attr6").WithDataType(entity.FieldTypeBool)).
		WithField(entity.NewField().WithName(testVectorField).WithDataType(entity.FieldTypeFloatVector).WithDim(testVectorDim)).
		WithField(entity.NewField().WithName("binary_vector").WithDataType(entity.FieldTypeBinaryVector).WithDim(testVectorDim))

	// one row
	columnID := entity.NewColumnInt64(testPrimaryField, []int64{0})
	columnAttr1 := entity.NewColumnInt8("attr1", []int8{0})
	columnAttr2 := entity.NewColumnInt16("attr2", []int16{0})
	columnAttr3 := entity.NewColumnInt32("attr3", []int32{0})
	columnAttr4 := entity.NewColumnFloat("attr4", []float32{0})
	columnAttr5 := entity.NewColumnDouble("attr5", []float64{0})
	columnAttr6 := entity.NewColumnBool("attr6", []bool{true})
	columnFv := entity.NewColumnFloatVector(testVectorField, testVectorDim, generateFloatVector(1, testVectorDim))
	columnBv := entity.NewColumnBinaryVector("binary_vector", testVectorDim, generateBinaryVector(1, testVectorDim))

	sr := &milvuspb.SearchResults{
		Results: &schemapb.SearchResultData{
			FieldsData: []*schemapb.FieldData{
				columnID.FieldData(),
				columnAttr1.FieldData(),
				columnAttr2.FieldData(),
				columnAttr3.FieldData(),
				columnAttr4.FieldData(),
				columnAttr5.FieldData(),
				columnAttr6.FieldData(),
				columnFv.FieldData(),
				columnBv.FieldData(),
			},
		},
	}
	bs, err := proto.Marshal(sr)
	assert.Nil(t, err)
	sr1l := len(bs)
	// 2Row
	columnID = entity.NewColumnInt64(testPrimaryField, []int64{0, 1})
	columnAttr1 = entity.NewColumnInt8("attr1", []int8{0, 1})
	columnAttr2 = entity.NewColumnInt16("attr2", []int16{0, 1})
	columnAttr3 = entity.NewColumnInt32("attr3", []int32{0, 1})
	columnAttr4 = entity.NewColumnFloat("attr4", []float32{0, 1})
	columnAttr5 = entity.NewColumnDouble("attr5", []float64{0, 1})
	columnAttr6 = entity.NewColumnBool("attr6", []bool{true, true})
	columnFv = entity.NewColumnFloatVector(testVectorField, testVectorDim, generateFloatVector(2, testVectorDim))
	columnBv = entity.NewColumnBinaryVector("binary_vector", testVectorDim, generateBinaryVector(2, testVectorDim))

	sr = &milvuspb.SearchResults{
		Results: &schemapb.SearchResultData{
			FieldsData: []*schemapb.FieldData{
				columnID.FieldData(),
				columnAttr1.FieldData(),
				columnAttr2.FieldData(),
				columnAttr3.FieldData(),
				columnAttr4.FieldData(),
				columnAttr5.FieldData(),
				columnAttr6.FieldData(),
				columnFv.FieldData(),
				columnBv.FieldData(),
			},
		},
	}
	bs, err = proto.Marshal(sr)
	assert.Nil(t, err)
	sr2l := len(bs)

	t.Log(sr1l, sr2l, sr2l-sr1l)
	est := estRowSize(sch, []string{})
	t.Log(est)

	assert.Greater(t, est, int64(sr2l-sr1l))
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

func generateBinaryVector(num, dim int) [][]byte {
	r := make([][]byte, 0, num)
	for i := 0; i < num; i++ {
		v := make([]byte, 0, dim/8)
		rand.Read(v)
		r = append(r, v)
	}
	return r
}

func TestVector2PlaceHolder(t *testing.T) {
	t.Run("FloatVector", func(t *testing.T) {
		data := generateFloatVector(10, 32)
		vectors := make([]entity.Vector, 0, len(data))
		for _, row := range data {
			vectors = append(vectors, entity.FloatVector(row))
		}

		phv := vector2Placeholder(vectors)
		assert.Equal(t, "$0", phv.Tag)
		assert.Equal(t, commonpb.PlaceholderType_FloatVector, phv.Type)
		require.Equal(t, len(vectors), len(phv.Values))
		for idx, line := range phv.Values {
			assert.Equal(t, vectors[idx].Serialize(), line)
		}
	})

	t.Run("BinaryVector", func(t *testing.T) {
		data := generateBinaryVector(10, 32)
		vectors := make([]entity.Vector, 0, len(data))
		for _, row := range data {
			vectors = append(vectors, entity.BinaryVector(row))
		}

		phv := vector2Placeholder(vectors)
		assert.Equal(t, "$0", phv.Tag)
		assert.Equal(t, commonpb.PlaceholderType_BinaryVector, phv.Type)
		require.Equal(t, len(vectors), len(phv.Values))
		for idx, line := range phv.Values {
			assert.Equal(t, vectors[idx].Serialize(), line)
		}
	})
}

type WildcardSuite struct {
	suite.Suite

	schema *entity.Schema
}

func (s *WildcardSuite) SetupTest() {
	s.schema = entity.NewSchema().
		WithField(entity.NewField().WithName("pk").WithDataType(entity.FieldTypeInt64).WithIsPrimaryKey(true)).
		WithField(entity.NewField().WithName("attr").WithDataType(entity.FieldTypeInt64)).
		WithField(entity.NewField().WithName("$meta").WithDataType(entity.FieldTypeJSON).WithIsDynamic(true)).
		WithField(entity.NewField().WithName("vector").WithDataType(entity.FieldTypeFloatVector).WithDim(128))
}

func (s *WildcardSuite) TestExpandWildcard() {
	type testCase struct {
		tag            string
		input          []string
		expect         []string
		expectWildCard bool
	}

	cases := []testCase{
		{tag: "normal", input: []string{"pk", "attr"}, expect: []string{"pk", "attr"}},
		{tag: "with_wildcard", input: []string{"*"}, expect: []string{"pk", "attr", "$meta", "vector"}, expectWildCard: true},
		{tag: "wildcard_dynamic", input: []string{"*", "a"}, expect: []string{"pk", "attr", "$meta", "vector", "a"}, expectWildCard: true},
	}

	for _, tc := range cases {
		s.Run(tc.tag, func() {
			output, wildCard := expandWildcard(s.schema, tc.input)
			s.ElementsMatch(tc.expect, output)
			s.Equal(tc.expectWildCard, wildCard)
		})
	}
}

func TestExpandWildcard(t *testing.T) {
	suite.Run(t, new(WildcardSuite))
}
