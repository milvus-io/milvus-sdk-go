package client

import (
	"context"
	"errors"
	"fmt"
	"math/rand"
	"testing"
	"time"

	"github.com/golang/protobuf/proto"
	common "github.com/milvus-io/milvus-proto/go-api/commonpb"
	server "github.com/milvus-io/milvus-proto/go-api/milvuspb"
	schema "github.com/milvus-io/milvus-proto/go-api/schemapb"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
	"github.com/stretchr/testify/assert"
)

func TestGrpcClientListCollections(t *testing.T) {
	ctx := context.Background()
	c := testClient(ctx, t)

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

	for _, tc := range cases {
		mock.SetInjection(MShowCollections, func(_ context.Context, raw proto.Message) (proto.Message, error) {
			s, err := SuccessStatus()
			resp := &server.ShowCollectionsResponse{
				Status:              s,
				CollectionIds:       tc.ids,
				CollectionNames:     tc.names,
				InMemoryPercentages: tc.inMem,
			}
			return resp, err
		})
		collections, err := c.ListCollections(ctx)
		if assert.Nil(t, err) && assert.Equal(t, tc.collNum, len(collections)) {
			// assert element match
			rids := make([]int64, 0, len(collections))
			rnames := make([]string, 0, len(collections))
			for _, collection := range collections {
				rids = append(rids, collection.ID)
				rnames = append(rnames, collection.Name)
			}
			assert.ElementsMatch(t, tc.ids, rids)
			assert.ElementsMatch(t, tc.names, rnames)
			// assert id & name match
			for idx, rid := range rids {
				for jdx, id := range tc.ids {
					if rid == id {
						assert.Equal(t, tc.names[jdx], rnames[idx])
						assert.Equal(t, tc.inMem[jdx] == 100, collections[idx].Loaded)
					}
				}
			}
		}
	}
}

func TestGrpcClientCreateCollection(t *testing.T) {
	ctx := context.Background()
	c := testClient(ctx, t)
	// default, all collection name returns false
	mock.DelInjection(MHasCollection)
	t.Run("Test normal creation", func(t *testing.T) {
		ds := defaultSchema()
		shardsNum := int32(1)
		mock.SetInjection(MCreateCollection, func(ctx context.Context, raw proto.Message) (proto.Message, error) {
			req, ok := raw.(*server.CreateCollectionRequest)
			if !ok {
				return &common.Status{ErrorCode: common.ErrorCode_IllegalArgument}, errors.New("illegal request type")
			}
			assert.Equal(t, testCollectionName, req.GetCollectionName())
			sschema := &schema.CollectionSchema{}
			if !assert.Nil(t, proto.Unmarshal(req.GetSchema(), sschema)) {
				if assert.Equal(t, len(ds.Fields), len(sschema.Fields)) {
					for idx, fieldSchema := range ds.Fields {
						assert.Equal(t, fieldSchema.Name, sschema.GetFields()[idx].GetName())
						assert.Equal(t, fieldSchema.PrimaryKey, sschema.GetFields()[idx].GetIsPrimaryKey())
						assert.Equal(t, fieldSchema.AutoID, sschema.GetFields()[idx].GetAutoID())
						assert.EqualValues(t, fieldSchema.DataType, sschema.GetFields()[idx].GetDataType())
					}
				}
				assert.Equal(t, shardsNum, req.GetShardsNum())
			}

			return &common.Status{ErrorCode: common.ErrorCode_Success}, nil
		})
		assert.Nil(t, c.CreateCollection(ctx, ds, shardsNum))
		mock.DelInjection(MCreateCollection)
	})

	t.Run("Test create with consistency level", func(t *testing.T) {
		ds := defaultSchema()
		shardsNum := int32(1)
		mock.SetInjection(MCreateCollection, func(ctx context.Context, raw proto.Message) (proto.Message, error) {
			req, ok := raw.(*server.CreateCollectionRequest)
			if !ok {
				return &common.Status{ErrorCode: common.ErrorCode_IllegalArgument}, errors.New("illegal request type")
			}
			assert.Equal(t, testCollectionName, req.GetCollectionName())
			assert.Equal(t, common.ConsistencyLevel_Eventually, req.GetConsistencyLevel())

			return &common.Status{ErrorCode: common.ErrorCode_Success}, nil
		})
		assert.Nil(t, c.CreateCollection(ctx, ds, shardsNum, WithConsistencyLevel(entity.ClEventually)))
		mock.DelInjection(MCreateCollection)
	})

	t.Run("Test invalid schemas", func(t *testing.T) {
		cases := []*entity.Schema{
			// empty fields
			{
				CollectionName: testCollectionName,
				Fields:         []*entity.Field{},
			},
			// empty collection name
			{
				CollectionName: "",
				Fields: []*entity.Field{
					{
						Name:       "int64",
						DataType:   entity.FieldTypeInt64,
						PrimaryKey: true,
					},
					{
						Name:       "vector",
						DataType:   entity.FieldTypeFloatVector,
						TypeParams: map[string]string{entity.TypeParamDim: "128"},
					},
				},
			},
			// multiple primary key
			{
				CollectionName: testCollectionName,
				Fields: []*entity.Field{
					{
						Name:       "int64",
						DataType:   entity.FieldTypeInt64,
						PrimaryKey: true,
					},
					{
						Name:       "int64_2",
						DataType:   entity.FieldTypeInt64,
						PrimaryKey: true,
					},
					{
						Name:       "vector",
						DataType:   entity.FieldTypeFloatVector,
						TypeParams: map[string]string{entity.TypeParamDim: "128"},
					},
				},
			},
			// multiple auto id
			{
				CollectionName: testCollectionName,
				Fields: []*entity.Field{
					{
						Name:       "int64",
						DataType:   entity.FieldTypeInt64,
						PrimaryKey: true,
						AutoID:     true,
					},
					{
						Name:       "int64_2",
						DataType:   entity.FieldTypeInt64,
						PrimaryKey: false,
						AutoID:     true,
					},
					{
						Name:       "vector",
						DataType:   entity.FieldTypeFloatVector,
						TypeParams: map[string]string{entity.TypeParamDim: "128"},
					},
				},
			},
			// Bad primary key type
			{
				CollectionName: testCollectionName,
				Fields: []*entity.Field{
					{
						Name:       "float_pk",
						DataType:   entity.FieldTypeFloat,
						PrimaryKey: true,
					},
					{
						Name:       "vector",
						DataType:   entity.FieldTypeFloatVector,
						TypeParams: map[string]string{entity.TypeParamDim: "128"},
					},
				},
			},
		}
		shardsNum := int32(1) // <= 0 will used default shards num 2, skip check
		mock.SetInjection(MCreateCollection, func(_ context.Context, _ proto.Message) (proto.Message, error) {
			// should not be here!
			assert.FailNow(t, "should not be here")
			return nil, errors.New("should not be here")
		})
		for _, s := range cases {
			assert.NotNil(t, c.CreateCollection(ctx, s, shardsNum))
			mock.DelInjection(MCreateCollection)
		}
	})

	t.Run("test duplicated collection", func(t *testing.T) {
		m := make(map[string]struct{})
		mock.SetInjection(MCreateCollection, func(_ context.Context, raw proto.Message) (proto.Message, error) {
			req, ok := raw.(*server.CreateCollectionRequest)
			if !ok {
				return BadRequestStatus()
			}
			m[req.GetCollectionName()] = struct{}{}

			return SuccessStatus()
		})
		defer mock.DelInjection(MCreateCollection)
		mock.SetInjection(MHasCollection, func(_ context.Context, raw proto.Message) (proto.Message, error) {
			req, ok := raw.(*server.HasCollectionRequest)
			resp := &server.BoolResponse{}
			if !ok {
				return BadRequestStatus()
			}

			_, has := m[req.GetCollectionName()]
			resp.Value = has
			s, err := SuccessStatus()
			resp.Status = s
			return resp, err
		})
		defer mock.DelInjection(MHasCollection)

		assert.Nil(t, c.CreateCollection(ctx, defaultSchema(), 1))
		assert.NotNil(t, c.CreateCollection(ctx, defaultSchema(), 1))
	})

	t.Run("test server returns error", func(t *testing.T) {
		mock.SetInjection(MCreateCollection, func(ctx context.Context, raw proto.Message) (proto.Message, error) {
			_, ok := raw.(*server.CreateCollectionRequest)
			if !ok {
				return BadRequestStatus()
			}
			return &common.Status{
				ErrorCode: common.ErrorCode_UnexpectedError,
				Reason:    "Service is not healthy",
			}, nil
		})
		assert.Error(t, c.CreateCollection(ctx, defaultSchema(), 1))

		mock.SetInjection(MCreateCollection, func(ctx context.Context, raw proto.Message) (proto.Message, error) {
			return &common.Status{}, errors.New("mocked grpc error")
		})

		assert.Error(t, c.CreateCollection(ctx, defaultSchema(), 1))
		mock.DelInjection(MCreateCollection)
	})
}

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

func TestGrpcClientDropCollection(t *testing.T) {
	ctx := context.Background()
	c := testClient(ctx, t)

	mock.SetInjection(MHasCollection, hasCollectionDefault)
	mock.SetInjection(MDropCollection, func(_ context.Context, raw proto.Message) (proto.Message, error) {
		req, ok := (raw).(*server.DropCollectionRequest)
		if !ok {
			return BadRequestStatus()
		}
		if req.GetCollectionName() != testCollectionName { // in mock server, assume testCollection exists only
			return BadRequestStatus()
		}
		return SuccessStatus()
	})

	t.Run("Test Normal drop", func(t *testing.T) {
		assert.Nil(t, c.DropCollection(ctx, testCollectionName))
	})

	t.Run("Test drop non-existing collection", func(t *testing.T) {
		assert.NotNil(t, c.DropCollection(ctx, "AAAAAAAAAANonExists"))
	})
}

func TestGrpcClientLoadCollection(t *testing.T) {
	ctx := context.Background()
	c := testClient(ctx, t)
	mock.SetInjection(MHasCollection, hasCollectionDefault)
	// injection check collection name equals
	mock.SetInjection(MLoadCollection, func(_ context.Context, raw proto.Message) (proto.Message, error) {
		req, ok := raw.(*server.LoadCollectionRequest)
		if !ok {
			return BadRequestStatus()
		}
		assert.Equal(t, testCollectionName, req.GetCollectionName())
		return SuccessStatus()
	})
	t.Run("Load collection normal async", func(t *testing.T) {
		assert.Nil(t, c.LoadCollection(ctx, testCollectionName, true))
	})
	t.Run("Load collection sync", func(t *testing.T) {

		loadTime := rand.Intn(500) + 500 // in milli seconds, 100~1000 milliseconds
		passed := false                  // ### flag variable
		start := time.Now()

		mock.SetInjection(MShowCollections, func(_ context.Context, raw proto.Message) (proto.Message, error) {
			req, ok := raw.(*server.ShowCollectionsRequest)
			r := &server.ShowCollectionsResponse{}
			if !ok || req == nil {
				s, err := BadRequestStatus()
				r.Status = s
				return r, err
			}
			s, err := SuccessStatus()
			r.Status = s
			r.CollectionIds = []int64{1}
			var perc int64
			if time.Since(start) > time.Duration(loadTime)*time.Millisecond {
				t.Log("passed")
				perc = 100
				passed = true
			}
			r.InMemoryPercentages = []int64{perc}
			return r, err
		})
		assert.Nil(t, c.LoadCollection(ctx, testCollectionName, false))
		assert.True(t, passed)

		start = time.Now()
		passed = false
		quickCtx, cancel := context.WithTimeout(ctx, 50*time.Millisecond)
		defer cancel()
		assert.NotNil(t, c.LoadCollection(quickCtx, testCollectionName, false))

		// remove injection
		mock.DelInjection(MShowCollections)
	})
	t.Run("Load default replica", func(t *testing.T) {
		mock.SetInjection(MLoadCollection, func(ctx context.Context, raw proto.Message) (proto.Message, error) {
			req, ok := raw.(*server.LoadCollectionRequest)
			if !ok {
				return BadRequestStatus()
			}
			assert.Equal(t, testDefaultReplicaNumber, req.GetReplicaNumber())
			assert.Equal(t, testCollectionName, req.GetCollectionName())
			return SuccessStatus()
		})
		defer mock.DelInjection(MLoadCollection)
		assert.Nil(t, c.LoadCollection(ctx, testCollectionName, true))
	})
	t.Run("Load multiple replica", func(t *testing.T) {
		mock.DelInjection(MLoadCollection)

		mock.SetInjection(MLoadCollection, func(ctx context.Context, raw proto.Message) (proto.Message, error) {
			req, ok := raw.(*server.LoadCollectionRequest)
			if !ok {
				return BadRequestStatus()
			}
			assert.Equal(t, testMultiReplicaNumber, req.GetReplicaNumber())
			assert.Equal(t, testCollectionName, req.GetCollectionName())
			return SuccessStatus()
		})
		defer mock.DelInjection(MLoadCollection)
		assert.Nil(t, c.LoadCollection(ctx, testCollectionName, true, WithReplicaNumber(testMultiReplicaNumber)))
	})
}

func TestReleaseCollection(t *testing.T) {
	ctx := context.Background()

	c := testClient(ctx, t)

	mock.SetInjection(MReleaseCollection, func(_ context.Context, raw proto.Message) (proto.Message, error) {
		req, ok := raw.(*server.ReleaseCollectionRequest)
		if !ok {
			return BadRequestStatus()
		}
		assert.Equal(t, testCollectionName, req.GetCollectionName())
		return SuccessStatus()
	})

	c.ReleaseCollection(ctx, testCollectionName)
}

func TestGrpcClientHasCollection(t *testing.T) {
	ctx := context.Background()

	c := testClient(ctx, t)

	mock.SetInjection(MHasCollection, func(_ context.Context, raw proto.Message) (proto.Message, error) {
		req, ok := raw.(*server.HasCollectionRequest)
		resp := &server.BoolResponse{}
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

// return injection asserts collection name matchs
// partition name request in partitionNames if flag is true
func hasCollectionInjection(t *testing.T, mustIn bool, collNames ...string) func(context.Context, proto.Message) (proto.Message, error) {
	return func(_ context.Context, raw proto.Message) (proto.Message, error) {
		req, ok := raw.(*server.HasCollectionRequest)
		resp := &server.BoolResponse{}
		if !ok {
			s, err := BadRequestStatus()
			resp.Status = s
			return resp, err
		}
		if mustIn {
			resp.Value = assert.Contains(t, collNames, req.GetCollectionName())
		} else {
			for _, pn := range collNames {
				if pn == req.GetCollectionName() {
					resp.Value = true
				}
			}
		}
		s, err := SuccessStatus()
		resp.Status = s
		return resp, err
	}
}

func describeCollectionInjection(t *testing.T, collID int64, collName string, sch *entity.Schema) func(_ context.Context, raw proto.Message) (proto.Message, error) {
	return func(_ context.Context, raw proto.Message) (proto.Message, error) {
		req, ok := raw.(*server.DescribeCollectionRequest)
		resp := &server.DescribeCollectionResponse{}
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

	mock.SetInjection(MDescribeCollection, describeCollectionInjection(t, collectionID, testCollectionName, defaultSchema()))

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

	mock.SetInjection(MGetCollectionStatistics, func(_ context.Context, raw proto.Message) (proto.Message, error) {
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

func TestGrpcClientGetReplicas(t *testing.T) {
	ctx := context.Background()
	c := testClient(ctx, t)

	replicaID := rand.Int63()
	nodeIds := []int64{1, 2, 3, 4}
	mock.SetInjection(MHasCollection, hasCollectionDefault)
	defer mock.DelInjection(MHasCollection)

	mock.SetInjection(MShowCollections, func(_ context.Context, raw proto.Message) (proto.Message, error) {
		s, err := SuccessStatus()
		resp := &server.ShowCollectionsResponse{
			Status:              s,
			CollectionIds:       []int64{testCollectionID},
			CollectionNames:     []string{testCollectionName},
			InMemoryPercentages: []int64{100},
		}
		return resp, err
	})
	defer mock.DelInjection(MShowCollections)

	mock.SetInjection(MGetReplicas, func(ctx context.Context, raw proto.Message) (proto.Message, error) {
		req, ok := raw.(*server.GetReplicasRequest)
		resp := &server.GetReplicasResponse{}
		if !ok {
			s, err := BadRequestStatus()
			resp.Status = s
			return resp, err
		}

		assert.Equal(t, testCollectionID, req.CollectionID)

		s, err := SuccessStatus()
		resp.Status = s
		resp.Replicas = []*server.ReplicaInfo{{
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

	t.Run("get replicas invalid name", func(t *testing.T) {
		_, err := c.GetReplicas(ctx, "invalid name")
		assert.Error(t, err)
	})

	t.Run("get replicas grpc error", func(t *testing.T) {
		mock.SetInjection(MGetReplicas, func(ctx context.Context, raw proto.Message) (proto.Message, error) {
			return &server.GetReplicasResponse{}, errors.New("mocked grpc error")
		})
		_, err := c.GetReplicas(ctx, testCollectionName)
		assert.Error(t, err)
	})

	t.Run("get replicas server error", func(t *testing.T) {
		mock.SetInjection(MGetReplicas, func(ctx context.Context, raw proto.Message) (proto.Message, error) {
			return &server.GetReplicasResponse{
				Status: &common.Status{
					ErrorCode: common.ErrorCode_UnexpectedError,
					Reason:    "Service is not healthy",
				},
				Replicas: nil,
			}, nil
		})
		_, err := c.GetReplicas(ctx, testCollectionName)
		assert.Error(t, err)
	})

	mock.DelInjection(MGetReplicas)
}

func TestGrpcClientGetLoadingProgress(t *testing.T) {
	ctx := context.Background()
	c := testClient(ctx, t)

	mock.SetInjection(MHasCollection, hasCollectionDefault)

	mock.SetInjection(MGetLoadingProgress, func(_ context.Context, raw proto.Message) (proto.Message, error) {
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

	mock.SetInjection(MHasCollection, hasCollectionDefault)

	mock.SetInjection(MGetLoadState, func(_ context.Context, raw proto.Message) (proto.Message, error) {
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
