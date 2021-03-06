package client

import (
	"context"
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
		ids, err = c.Insert(ctx, testCollectionName, "", entity.NewColumnInt64("int64", []int64{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}),
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
