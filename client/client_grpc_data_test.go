package client

import (
	"context"
	"errors"
	"fmt"
	"math/rand"
	"testing"
	"time"

	"github.com/golang/protobuf/proto"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
	"github.com/milvus-io/milvus-sdk-go/v2/internal/proto/common"
	"github.com/milvus-io/milvus-sdk-go/v2/internal/proto/schema"
	"github.com/milvus-io/milvus-sdk-go/v2/internal/proto/server"
	"github.com/stretchr/testify/assert"
)

func TestGrpcClientInsert(t *testing.T) {
	ctx := context.Background()

	c := testClient(ctx, t)

	t.Run("test create failure due to meta", func(t *testing.T) {
		mock.delInjection(mHasCollection) // collection does not exist
		ids, err := c.Insert(ctx, testCollectionName, "")
		assert.Nil(t, ids)
		assert.NotNil(t, err)

		// partition not exists
		mock.setInjection(mHasCollection, hasCollectionDefault)
		ids, err = c.Insert(ctx, testCollectionName, "_part_not_exists")
		assert.Nil(t, ids)
		assert.NotNil(t, err)

		// field not in collection
		mock.setInjection(mDescribeCollection, describeCollectionInjection(t, 0, testCollectionName, defaultSchema()))
		vectors := generateFloatVector(10, testVectorDim)
		ids, err = c.Insert(ctx, testCollectionName, "",
			entity.NewColumnInt64("extra_field", []int64{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}), entity.NewColumnFloatVector(testVectorField, testVectorDim, vectors))
		assert.Nil(t, ids)
		assert.NotNil(t, err)

		// field type not match
		mock.setInjection(mDescribeCollection, describeCollectionInjection(t, 0, testCollectionName, defaultSchema()))
		ids, err = c.Insert(ctx, testCollectionName, "",
			entity.NewColumnInt32("int64", []int32{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}), entity.NewColumnFloatVector(testVectorField, testVectorDim, vectors))
		assert.Nil(t, ids)
		assert.NotNil(t, err)

		// missing field
		ids, err = c.Insert(ctx, testCollectionName, "")
		assert.Nil(t, ids)
		assert.NotNil(t, err)

		// column len not match
		ids, err = c.Insert(ctx, testCollectionName, "", entity.NewColumnInt64("int64", []int64{1, 2, 3, 4, 5, 6, 7, 8, 9}),
			entity.NewColumnFloatVector(testVectorField, testVectorDim, vectors))
		assert.Nil(t, ids)
		assert.NotNil(t, err)

		// column len not match
		ids, err = c.Insert(ctx, testCollectionName, "", entity.NewColumnInt64("int64", []int64{1, 2, 3}),
			entity.NewColumnFloatVector(testVectorField, testVectorDim, vectors))
		assert.Nil(t, ids)
		assert.NotNil(t, err)

		// dim not match
		ids, err = c.Insert(ctx, testCollectionName, "", entity.NewColumnInt64("int64", []int64{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}),
			entity.NewColumnFloatVector(testVectorField, testVectorDim*2, vectors))
		assert.Nil(t, ids)
		assert.NotNil(t, err)
	})

	mock.setInjection(mHasCollection, hasCollectionDefault)
	mock.setInjection(mDescribeCollection, describeCollectionInjection(t, 0, testCollectionName, defaultSchema()))

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
	mock.delInjection(mInsert)
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
		// 510ms ~ 2s
		flushTime := 510 + rand.Intn(1500)
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

		mock.setInjection(mGetFlushState, func(_ context.Context, raw proto.Message) (proto.Message, error) {
			req, ok := raw.(*server.GetFlushStateRequest)
			resp := &server.GetFlushStateResponse{}
			if !ok {
				s, err := badRequestStatus()
				resp.Status = s
				return resp, err
			}
			assert.ElementsMatch(t, segments, req.GetSegmentIDs())
			resp.Flushed = false
			if time.Since(start) > time.Duration(flushTime)*time.Millisecond {
				resp.Flushed = true
				flag = true
			}

			s, err := successStatus()
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

func TestGrpcDeleteByPks(t *testing.T) {
	ctx := context.Background()

	c := testClient(ctx, t)
	defer c.Close()

	mock.setInjection(mDescribeCollection, describeCollectionInjection(t, 1, testCollectionName, defaultSchema()))
	defer mock.delInjection(mDescribeCollection)

	t.Run("normal delete by pks", func(t *testing.T) {
		partName := "testPart"
		mock.setInjection(mHasPartition, hasPartitionInjection(t, testCollectionName, true, partName))
		defer mock.delInjection(mHasPartition)
		mock.setInjection(mDelete, func(_ context.Context, raw proto.Message) (proto.Message, error) {
			req, ok := raw.(*server.DeleteRequest)
			if !ok {
				t.FailNow()
			}
			assert.Equal(t, testCollectionName, req.GetCollectionName())
			assert.Equal(t, partName, req.GetPartitionName())

			resp := &server.MutationResult{}
			s, err := successStatus()
			resp.Status = s
			return resp, err
		})
		defer mock.delInjection(mDelete)

		err := c.DeleteByPks(ctx, testCollectionName, partName, entity.NewColumnInt64(testPrimaryField, []int64{1, 2, 3}))
		assert.NoError(t, err)
	})

	t.Run("Bad request deletes", func(t *testing.T) {
		partName := "testPart"
		mock.setInjection(mHasPartition, hasPartitionInjection(t, testCollectionName, false, partName))
		defer mock.delInjection(mHasPartition)

		// non-exist collection
		err := c.DeleteByPks(ctx, "non-exists-collection", "", entity.NewColumnInt64("pk", []int64{}))
		assert.Error(t, err)

		// non-exist parition
		err = c.DeleteByPks(ctx, testCollectionName, "non-exists-part", entity.NewColumnInt64("pk", []int64{}))
		assert.Error(t, err)

		// zero length pk
		err = c.DeleteByPks(ctx, testCollectionName, "", entity.NewColumnInt64(testPrimaryField, []int64{}))
		assert.Error(t, err)

		// string pk field
		err = c.DeleteByPks(ctx, testCollectionName, "", entity.NewColumnString(testPrimaryField, []string{"1"}))
		assert.Error(t, err)

		// pk name not match
		err = c.DeleteByPks(ctx, testCollectionName, "", entity.NewColumnInt64("not_pk", []int64{1}))
		assert.Error(t, err)
	})

	t.Run("delete services fail", func(t *testing.T) {
		mock.setInjection(mDelete, func(_ context.Context, raw proto.Message) (proto.Message, error) {
			resp := &server.MutationResult{}
			return resp, errors.New("mocked error")
		})

		err := c.DeleteByPks(ctx, testCollectionName, "", entity.NewColumnInt64(testPrimaryField, []int64{1}))
		assert.Error(t, err)

		mock.setInjection(mDelete, func(_ context.Context, raw proto.Message) (proto.Message, error) {
			resp := &server.MutationResult{}
			resp.Status = &common.Status{
				ErrorCode: common.ErrorCode_UnexpectedError,
			}
			return resp, nil
		})
		err = c.DeleteByPks(ctx, testCollectionName, "", entity.NewColumnInt64(testPrimaryField, []int64{1}))
		assert.Error(t, err)
	})
}

func TestGrpcSearch(t *testing.T) {

	ctx := context.Background()

	c := testClient(ctx, t)
	defer c.Close()
	vectors := generateFloatVector(4096, testVectorDim)

	t.Run("search fail due to meta error", func(t *testing.T) {
		sp, err := entity.NewIndexFlatSearchParam(10)
		assert.Nil(t, err)

		// collection name
		mock.delInjection(mHasCollection)
		r, err := c.Search(ctx, testCollectionName, []string{}, "", []string{}, []entity.Vector{}, "vector",
			entity.L2, 5, sp)
		assert.Nil(t, r)
		assert.NotNil(t, err)

		// partition
		mock.setInjection(mHasCollection, hasCollectionDefault)
		r, err = c.Search(ctx, testCollectionName, []string{"_non_exist"}, "", []string{}, []entity.Vector{}, "vector",
			entity.L2, 5, sp)
		assert.Nil(t, r)
		assert.NotNil(t, err)

		// output field
		mock.setInjection(mDescribeCollection, describeCollectionInjection(t, 0, testCollectionName, defaultSchema()))
		r, err = c.Search(ctx, testCollectionName, []string{}, "", []string{"extra"}, []entity.Vector{}, "vector",
			entity.L2, 5, sp)
		assert.Nil(t, r)
		assert.NotNil(t, err)

		// vector field
		mock.setInjection(mDescribeCollection, describeCollectionInjection(t, 0, testCollectionName, defaultSchema()))
		r, err = c.Search(ctx, testCollectionName, []string{}, "", []string{"int64"}, []entity.Vector{}, "no_vector",
			entity.L2, 5, sp)
		assert.Nil(t, r)
		assert.NotNil(t, err)

		// vector dim
		badVectors := generateFloatVector(1, testVectorDim*2)
		r, err = c.Search(ctx, testCollectionName, []string{}, "", []string{"int64"}, []entity.Vector{entity.FloatVector(badVectors[0])}, "vector",
			entity.L2, 5, sp)
		assert.Nil(t, r)
		assert.NotNil(t, err)

		// metric type
		r, err = c.Search(ctx, testCollectionName, []string{}, "", []string{"int64"}, []entity.Vector{entity.FloatVector(vectors[0])}, "vector",
			entity.HAMMING, 5, sp)
		assert.Nil(t, r)
		assert.NotNil(t, err)

	})

	t.Run("ok search", func(t *testing.T) {
		mock.setInjection(mHasCollection, hasCollectionDefault)
		mock.setInjection(mDescribeCollection, describeCollectionInjection(t, 0, testCollectionName, defaultSchema()))

		expr := `int64 > 0`

		mock.setInjection(mSearch, func(_ context.Context, raw proto.Message) (proto.Message, error) {
			req, ok := raw.(*server.SearchRequest)
			resp := &server.SearchResults{}
			if !ok {
				s, err := badRequestStatus()
				resp.Status = s
				return resp, err
			}
			assert.Equal(t, testCollectionName, req.GetCollectionName())
			assert.Equal(t, expr, req.GetDsl())
			assert.Equal(t, common.DslType_BoolExprV1, req.GetDslType())
			assert.ElementsMatch(t, []string{"int64"}, req.GetOutputFields())

			resp.Results = &schema.SearchResultData{
				NumQueries: 1,
				TopK:       10,
				FieldsData: []*schema.FieldData{
					{
						Type:      schema.DataType_Int64,
						FieldName: "int64",
						Field: &schema.FieldData_Scalars{
							Scalars: &schema.ScalarField{
								Data: &schema.ScalarField_LongData{
									LongData: &schema.LongArray{
										Data: []int64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
									},
								},
							},
						},
					},
				},
				Ids: &schema.IDs{
					IdField: &schema.IDs_IntId{
						IntId: &schema.LongArray{
							Data: []int64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
						},
					},
				},
				Scores: make([]float32, 10),
				Topks:  []int64{10},
			}

			s, err := successStatus()
			resp.Status = s
			return resp, err
		})

		sp, err := entity.NewIndexFlatSearchParam(10)
		assert.Nil(t, err)
		assert.NotNil(t, sp)
		results, err := c.Search(ctx, testCollectionName, []string{}, expr, []string{"int64"}, []entity.Vector{entity.FloatVector(vectors[0])},
			testVectorField, entity.L2, 10, sp)

		assert.Nil(t, err)
		assert.NotNil(t, results)
	})
}

func TestGrpcQueryByPks(t *testing.T) {
	ctx := context.Background()
	c := testClient(ctx, t)
	partName := "testPart"

	t.Run("normal query by pks", func(t *testing.T) {
		mock.setInjection(mHasCollection, hasCollectionDefault)
		mock.setInjection(mHasPartition, hasPartitionInjection(t, testCollectionName, false, partName))
		defer mock.delInjection(mHasPartition)

		mock.setInjection(mQuery, func(_ context.Context, raw proto.Message) (proto.Message, error) {
			req, ok := raw.(*server.QueryRequest)
			if !ok {
				t.FailNow()
			}
			assert.Equal(t, testCollectionName, req.GetCollectionName())
			if len(req.GetPartitionNames()) > 0 {
				assert.Equal(t, partName, req.GetPartitionNames()[0])
			}

			resp := &server.QueryResults{}
			s, err := successStatus()
			resp.Status = s
			resp.FieldsData = []*schema.FieldData{
				{
					Type:      schema.DataType_Int64,
					FieldName: "int64",
					Field: &schema.FieldData_Scalars{
						Scalars: &schema.ScalarField{
							Data: &schema.ScalarField_LongData{
								LongData: &schema.LongArray{
									Data: []int64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
								},
							},
						},
					},
				},
				{
					Type:      schema.DataType_FloatVector,
					FieldName: testVectorField,
					Field: &schema.FieldData_Vectors{
						Vectors: &schema.VectorField{
							Dim: 1,
							Data: &schema.VectorField_FloatVector{
								FloatVector: &schema.FloatArray{
									Data: []float32{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0},
								},
							},
						},
					},
				},
			}

			return resp, err
		})
		defer mock.delInjection(mQuery)

		columns, err := c.QueryByPks(ctx, testCollectionName, []string{partName}, entity.NewColumnInt64(testPrimaryField, []int64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}), []string{"int64", testVectorField})
		assert.NoError(t, err)
		assert.Equal(t, 2, len(columns))
		assert.Equal(t, entity.FieldTypeInt64, columns[0].Type())
		assert.Equal(t, entity.FieldTypeFloatVector, columns[1].Type())
		assert.Equal(t, 10, columns[0].Len())

		colInt64, ok := columns[0].(*entity.ColumnInt64)
		assert.True(t, ok)
		assert.ElementsMatch(t, []int64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, colInt64.Data())

	})

	t.Run("Bad request querys", func(t *testing.T) {
		mock.setInjection(mHasCollection, hasCollectionDefault)
		mock.setInjection(mHasPartition, hasPartitionInjection(t, testCollectionName, false, partName))
		defer mock.delInjection(mHasPartition)

		// non-exist collection
		_, err := c.QueryByPks(ctx, "non-exists-collection", []string{}, entity.NewColumnInt64("pk", []int64{}), []string{})
		assert.Error(t, err)

		// non-exist parition
		_, err = c.QueryByPks(ctx, testCollectionName, []string{"non-exists-part"}, entity.NewColumnInt64("pk", []int64{}), []string{})
		assert.Error(t, err)

		// zero length pk
		_, err = c.QueryByPks(ctx, testCollectionName, []string{}, entity.NewColumnInt64("pk", []int64{}), []string{})
		assert.Error(t, err)

		// string pk field
		_, err = c.QueryByPks(ctx, testCollectionName, []string{}, entity.NewColumnString("pk", []string{"1"}), []string{})
		assert.Error(t, err)

		// pk name not match
		_, err = c.QueryByPks(ctx, testCollectionName, []string{}, entity.NewColumnInt64("non_pk", []int64{1}), []string{})
		assert.Error(t, err)
	})

	t.Run("Query service error", func(t *testing.T) {
		mock.setInjection(mHasCollection, hasCollectionDefault)
		mock.setInjection(mHasPartition, hasPartitionInjection(t, testCollectionName, false, partName))
		defer mock.delInjection(mHasPartition)

		mock.setInjection(mQuery, func(_ context.Context, raw proto.Message) (proto.Message, error) {
			_, ok := raw.(*server.QueryRequest)
			if !ok {
				t.FailNow()
			}

			resp := &server.QueryResults{}
			return resp, errors.New("mocked error")
		})
		defer mock.delInjection(mQuery)

		_, err := c.QueryByPks(ctx, testCollectionName, []string{}, entity.NewColumnInt64(testPrimaryField, []int64{1}), []string{"*"})
		assert.Error(t, err)

		mock.setInjection(mQuery, func(_ context.Context, raw proto.Message) (proto.Message, error) {
			_, ok := raw.(*server.QueryRequest)
			if !ok {
				t.FailNow()
			}

			resp := &server.QueryResults{}
			resp.Status = &common.Status{
				ErrorCode: common.ErrorCode_UnexpectedError,
				Reason:    "mocked error",
			}
			return resp, nil
		})

		_, err = c.QueryByPks(ctx, testCollectionName, []string{}, entity.NewColumnInt64(testPrimaryField, []int64{1}), []string{"*"})
		assert.Error(t, err)

		mock.setInjection(mQuery, func(_ context.Context, raw proto.Message) (proto.Message, error) {
			_, ok := raw.(*server.QueryRequest)
			if !ok {
				t.FailNow()
			}

			resp := &server.QueryResults{}
			s, err := successStatus()
			resp.Status = s
			resp.FieldsData = []*schema.FieldData{
				{
					Type:      schema.DataType_String,
					FieldName: "int64",
					Field: &schema.FieldData_Scalars{
						Scalars: &schema.ScalarField{
							Data: &schema.ScalarField_LongData{
								LongData: &schema.LongArray{
									Data: []int64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
								},
							},
						},
					},
				},
			}

			return resp, err
		})
		_, err = c.QueryByPks(ctx, testCollectionName, []string{}, entity.NewColumnInt64(testPrimaryField, []int64{1}), []string{"*"})
		assert.Error(t, err)

		mock.setInjection(mQuery, func(_ context.Context, raw proto.Message) (proto.Message, error) {
			_, ok := raw.(*server.QueryRequest)
			if !ok {
				t.FailNow()
			}

			resp := &server.QueryResults{}
			s, err := successStatus()
			resp.Status = s
			resp.FieldsData = []*schema.FieldData{
				{
					Type:      schema.DataType_FloatVector,
					FieldName: "int64",
					Field: &schema.FieldData_Scalars{
						Scalars: &schema.ScalarField{
							Data: &schema.ScalarField_LongData{
								LongData: &schema.LongArray{
									Data: []int64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
								},
							},
						},
					},
				},
			}

			return resp, err
		})
		_, err = c.QueryByPks(ctx, testCollectionName, []string{}, entity.NewColumnInt64(testPrimaryField, []int64{1}), []string{"*"})
		assert.Error(t, err)
	})
}

func TestGrpcCalcDistanceWithIDs(t *testing.T) {
	ctx := context.Background()
	t.Run("bad client calls CalcDistance", func(t *testing.T) {
		c := &grpcClient{}
		r, err := c.CalcDistance(ctx, testCollectionName, []string{}, entity.L2, nil, nil)
		assert.Nil(t, r)
		assert.NotNil(t, err)
		assert.EqualValues(t, ErrClientNotReady, err)
	})

	c := testClient(ctx, t)
	mock.setInjection(mDescribeCollection, func(_ context.Context, raw proto.Message) (proto.Message, error) {
		req, ok := raw.(*server.DescribeCollectionRequest)
		resp := &server.DescribeCollectionResponse{}
		if !ok {
			s, err := badRequestStatus()
			resp.Status = s
			return resp, err
		}
		assert.Equal(t, testCollectionName, req.GetCollectionName())

		sch := defaultSchema()
		resp.Schema = sch.ProtoMessage()

		s, err := successStatus()
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
		mock.setInjection(mCalcDistance, func(_ context.Context, raw proto.Message) (proto.Message, error) {
			req, ok := raw.(*server.CalcDistanceRequest)
			resp := &server.CalcDistanceResults{}
			if !ok {
				s, err := badRequestStatus()
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
				dl = len(idsLeft.IdArray.GetIntId().Data)
			}
			if valuesLeft != nil {
				dl = len(valuesLeft.GetFloatVector().GetData()) / int(valuesLeft.Dim)
			}
			dr := 0
			if idsRight != nil {
				dr = len(idsRight.IdArray.GetIntId().Data)
			}
			if valuesRight != nil {
				dr = len(valuesRight.GetFloatVector().GetData()) / int(valuesRight.Dim)
			}

			resp.Array = &server.CalcDistanceResults_FloatDist{
				FloatDist: &schema.FloatArray{
					Data: make([]float32, dl*dr),
				},
			}

			s, err := successStatus()
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
		mock.setInjection(mCalcDistance, func(_ context.Context, raw proto.Message) (proto.Message, error) {
			req, ok := raw.(*server.CalcDistanceRequest)
			resp := &server.CalcDistanceResults{}
			if !ok {
				s, err := badRequestStatus()
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
				dl = len(idsLeft.IdArray.GetIntId().Data)
			}
			if valuesLeft != nil {
				dl = len(valuesLeft.GetFloatVector().GetData()) / int(valuesLeft.Dim)
			}
			dr := 0
			if idsRight != nil {
				dr = len(idsRight.IdArray.GetIntId().Data)
			}
			if valuesRight != nil {
				dr = len(valuesRight.GetFloatVector().GetData()) / int(valuesRight.Dim)
			}

			resp.Array = &server.CalcDistanceResults_IntDist{
				IntDist: &schema.IntArray{
					Data: make([]int32, dl*dr),
				},
			}

			s, err := successStatus()
			resp.Status = s
			return resp, err
		})
		r, err = c.CalcDistance(ctx, testCollectionName, []string{}, entity.HAMMING,
			entity.NewColumnInt64("vector", []int64{1}), entity.NewColumnInt64("vector", []int64{1}))
		assert.Nil(t, err)
		assert.NotNil(t, r)

		// test str id
		mock.setInjection(mDescribeCollection, func(_ context.Context, raw proto.Message) (proto.Message, error) {
			req, ok := raw.(*server.DescribeCollectionRequest)
			resp := &server.DescribeCollectionResponse{}
			if !ok {
				s, err := badRequestStatus()
				resp.Status = s
				return resp, err
			}
			assert.Equal(t, testCollectionName, req.GetCollectionName())

			sch := defaultSchema()
			sch.Fields[0].DataType = entity.FieldTypeString
			sch.Fields[0].Name = "str"
			resp.Schema = sch.ProtoMessage()

			s, err := successStatus()
			resp.Status = s
			return resp, err
		})
		mock.setInjection(mCalcDistance, func(_ context.Context, raw proto.Message) (proto.Message, error) {
			req, ok := raw.(*server.CalcDistanceRequest)
			resp := &server.CalcDistanceResults{}
			if !ok {
				s, err := badRequestStatus()
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
			dl := len(idsLeft.IdArray.GetStrId().Data)

			resp.Array = &server.CalcDistanceResults_FloatDist{
				FloatDist: &schema.FloatArray{
					Data: make([]float32, dl),
				},
			}

			s, err := successStatus()
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
	sch := &entity.Schema{
		CollectionName: testCollectionName,
		AutoID:         false,
		Fields: []*entity.Field{
			{
				Name:       testPrimaryField,
				DataType:   entity.FieldTypeInt64,
				PrimaryKey: true,
				AutoID:     true,
			},
			{
				Name:     "attr1",
				DataType: entity.FieldTypeInt8,
			},
			{
				Name:     "attr2",
				DataType: entity.FieldTypeInt16,
			},
			{
				Name:     "attr3",
				DataType: entity.FieldTypeInt32,
			},
			{
				Name:     "attr4",
				DataType: entity.FieldTypeFloat,
			},
			{
				Name:     "attr5",
				DataType: entity.FieldTypeDouble,
			},
			{
				Name:     "attr6",
				DataType: entity.FieldTypeBool,
			},
			{
				Name:     testVectorField,
				DataType: entity.FieldTypeFloatVector,
				TypeParams: map[string]string{
					entity.TYPE_PARAM_DIM: fmt.Sprintf("%d", testVectorDim),
				},
			},
			{
				Name:     "binary_vector",
				DataType: entity.FieldTypeBinaryVector,
				TypeParams: map[string]string{
					entity.TYPE_PARAM_DIM: fmt.Sprintf("%d", testVectorDim),
				},
			},
		},
	}

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

	sr := &server.SearchResults{
		Results: &schema.SearchResultData{
			FieldsData: []*schema.FieldData{
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

	sr = &server.SearchResults{
		Results: &schema.SearchResultData{
			FieldsData: []*schema.FieldData{
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
