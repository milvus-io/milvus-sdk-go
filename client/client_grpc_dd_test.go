package client

import (
	"context"
	"errors"
	"fmt"
	"math/rand"
	"reflect"
	"testing"
	"time"

	"github.com/golang/protobuf/proto"
	"github.com/milvus-io/milvus-sdk-go/entity"
	"github.com/milvus-io/milvus-sdk-go/internal/proto/common"
	"github.com/milvus-io/milvus-sdk-go/internal/proto/schema"
	"github.com/milvus-io/milvus-sdk-go/internal/proto/server"
	"github.com/stretchr/testify/assert"
	"google.golang.org/grpc"
)

func TestGrpcClientNil(t *testing.T) {
	c := &grpcClient{}
	tp := reflect.TypeOf(c)
	v := reflect.ValueOf(c)
	ctx := context.Background()
	c2 := testClient(ctx, t)
	v2 := reflect.ValueOf(c2)

	ctxDone, cancel := context.WithCancel(context.Background())
	cancel() // cancel here, so the ctx is done already

	for i := 0; i < tp.NumMethod(); i++ {
		m := tp.Method(i)
		mt := m.Type                                   // type of function
		if m.Name == "Close" || m.Name == "Connect" || // skip connect & close
			m.Name == "Search" || // type alias MetricType treated as string
			m.Name == "Insert" { // complex methods with ...
			continue
		}
		ins := make([]reflect.Value, 0, mt.NumIn())
		for j := 1; j < mt.NumIn(); j++ { // idx == 0, is the receiver v
			if j == 1 {
				//non-general solution, hard code context!
				ins = append(ins, reflect.ValueOf(ctx))
				continue
			}
			inT := mt.In(j)

			switch inT.Kind() {
			case reflect.String: // pass empty
				ins = append(ins, reflect.ValueOf(""))
			case reflect.Int, reflect.Int64:
				ins = append(ins, reflect.ValueOf(0))
			case reflect.Bool:
				ins = append(ins, reflect.ValueOf(false))
			case reflect.Interface:
				idxType := reflect.TypeOf((*entity.Index)(nil)).Elem()
				if inT.Implements(idxType) {
					ins = append(ins, reflect.ValueOf(entity.NewFlatIndex("flat_index", entity.L2)))
				}
			default:
				ins = append(ins, reflect.Zero(inT))
			}
		}
		outs := v.MethodByName(m.Name).Call(ins)
		assert.True(t, len(outs) > 0)
		assert.EqualValues(t, ErrClientNotReady, outs[len(outs)-1].Interface())

		// ctx done

		if len(ins) > 0 { // with context param
			ins[0] = reflect.ValueOf(ctxDone)
			outs := v2.MethodByName(m.Name).Call(ins)
			assert.True(t, len(outs) > 0)
			assert.False(t, outs[len(outs)-1].IsNil())
		}
	}
}

func TestGrpcClientConnect(t *testing.T) {
	ctx := context.Background()

	t.Run("Use bufconn dailer, testing case", func(t *testing.T) {
		c, err := NewGrpcClient(ctx, "bufnet", grpc.WithBlock(), grpc.WithInsecure(), grpc.WithContextDialer(bufDialer)) // uses in memory buf connection
		assert.Nil(t, err)
		assert.NotNil(t, c)
	})

	t.Run("Test empty addr, using default timeout", func(t *testing.T) {
		c, err := NewGrpcClient(ctx, "")
		assert.NotNil(t, err)
		assert.Nil(t, c)
	})
}

func TestGrpcClientClose(t *testing.T) {
	ctx := context.Background()

	t.Run("normal close", func(t *testing.T) {
		c := testClient(ctx, t)
		assert.Nil(t, c.Close())
	})

	t.Run("double close", func(t *testing.T) {
		c := testClient(ctx, t)
		assert.Nil(t, c.Close())
		assert.Nil(t, c.Close())
	})
}

func TestGrpcClientListCollections(t *testing.T) {
	ctx := context.Background()
	c := testClient(ctx, t)

	type testCase struct {
		ids     []int64
		names   []string
		collNum int
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
		}
		cases = append(cases, tc)
	}

	for _, tc := range cases {
		mock.setInjection(mListCollection, func(_ context.Context, raw proto.Message) (proto.Message, error) {
			s, err := successStatus()
			resp := &server.ShowCollectionsResponse{
				Status:          s,
				CollectionIds:   tc.ids,
				CollectionNames: tc.names,
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
	mock.delInjection(mHasCollection)
	t.Run("Test normal creation", func(t *testing.T) {
		ds := defaultSchema()
		shardsNum := int32(1)
		mock.setInjection(mCreateCollection, func(ctx context.Context, raw proto.Message) (proto.Message, error) {
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
		mock.delInjection(mCreateCollection)
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
						TypeParams: map[string]string{"dim": "128"},
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
						TypeParams: map[string]string{"dim": "128"},
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
						TypeParams: map[string]string{"dim": "128"},
					},
				},
			},
		}
		shardsNum := int32(1) // <= 0 will used default shards num 2, skip check
		mock.setInjection(mCreateCollection, func(_ context.Context, _ proto.Message) (proto.Message, error) {
			// should not be here!
			assert.FailNow(t, "should not be here")
			return nil, errors.New("should not be here")
		})
		for _, s := range cases {
			assert.NotNil(t, c.CreateCollection(ctx, s, shardsNum))
			mock.delInjection(mCreateCollection)
		}
	})

	t.Run("test duplicated collection", func(t *testing.T) {
		m := make(map[string]struct{})
		mock.setInjection(mCreateCollection, func(_ context.Context, raw proto.Message) (proto.Message, error) {
			req, ok := raw.(*server.CreateCollectionRequest)
			if !ok {
				return badRequestStatus()
			}
			m[req.GetCollectionName()] = struct{}{}

			return successStatus()
		})
		mock.setInjection(mHasCollection, func(_ context.Context, raw proto.Message) (proto.Message, error) {
			req, ok := raw.(*server.HasCollectionRequest)
			resp := &server.BoolResponse{}
			if !ok {
				return badRequestStatus()
			}

			_, has := m[req.GetCollectionName()]
			resp.Value = has
			s, err := successStatus()
			resp.Status = s
			return resp, err
		})

		assert.Nil(t, c.CreateCollection(ctx, defaultSchema(), 1))
		assert.NotNil(t, c.CreateCollection(ctx, defaultSchema(), 1))
	})
}

// default HasCollection injection, returns true only when collection name is `testCollectionName`
var hasCollectionDefault = func(_ context.Context, raw proto.Message) (proto.Message, error) {
	req, ok := raw.(*server.HasCollectionRequest)
	resp := &server.BoolResponse{}
	if !ok {
		s, err := badRequestStatus()
		resp.Status = s
		return s, err
	}
	resp.Value = req.GetCollectionName() == testCollectionName
	s, err := successStatus()
	resp.Status = s
	return resp, err
}

func TestGrpcClientDropCollection(t *testing.T) {
	ctx := context.Background()
	c := testClient(ctx, t)

	mock.setInjection(mHasCollection, hasCollectionDefault)
	mock.setInjection(mDropCollection, func(_ context.Context, raw proto.Message) (proto.Message, error) {
		req, ok := (raw).(*server.DropCollectionRequest)
		if !ok {
			return badRequestStatus()
		}
		if req.GetCollectionName() != testCollectionName { // in mock server, assume testCollection exists only
			return badRequestStatus()
		}
		return successStatus()
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
	// injection check collection name equals
	mock.setInjection(mLoadCollection, func(_ context.Context, raw proto.Message) (proto.Message, error) {
		req, ok := raw.(*server.LoadCollectionRequest)
		if !ok {
			return badRequestStatus()
		}
		assert.Equal(t, testCollectionName, req.GetCollectionName())
		return successStatus()
	})
	t.Run("Load collection normal async", func(t *testing.T) {
		assert.Nil(t, c.LoadCollection(ctx, testCollectionName, true))
	})
	t.Run("Load collection sync", func(t *testing.T) {

		segmentCount := rand.Intn(10) + 1
		rowCounts := rand.Intn(20000) + 1
		loadTime := rand.Intn(900) + 100 // in milli seconds, 100~1000 milliseconds
		ok := false                      //### flag variable
		start := time.Now()

		mock.setInjection(mGetPersistentSegmentInfo, func(_ context.Context, raw proto.Message) (proto.Message, error) {
			s, err := successStatus()
			r := &server.GetPersistentSegmentInfoResponse{
				Status: s,
				Infos:  make([]*server.PersistentSegmentInfo, 0, segmentCount),
			}
			for i := 0; i < segmentCount; i++ {
				r.Infos = append(r.Infos, &server.PersistentSegmentInfo{
					SegmentID: int64(i),
					NumRows:   int64(rowCounts),
				})
			}
			r.Infos = append(r.Infos, &server.PersistentSegmentInfo{
				SegmentID: int64(segmentCount),
				NumRows:   0, // handcrafted empty segment
			})
			return r, err
		})
		mock.setInjection(mGetQuerySegmentInfo, func(_ context.Context, raw proto.Message) (proto.Message, error) {
			s, err := successStatus()
			r := &server.GetQuerySegmentInfoResponse{
				Status: s,
				Infos:  make([]*server.QuerySegmentInfo, 0, segmentCount),
			}
			rc := 0
			if time.Since(start) > time.Duration(loadTime)*time.Millisecond {
				rc = rowCounts // after load time, row counts set to full amount
				ok = true
			}
			for i := 0; i < segmentCount; i++ {
				r.Infos = append(r.Infos, &server.QuerySegmentInfo{
					SegmentID: int64(i),
					NumRows:   int64(rc),
				})
			}
			return r, err
		})

		assert.Nil(t, c.LoadCollection(ctx, testCollectionName, false))
		assert.True(t, ok)

		// remove injection
		mock.delInjection(mGetPersistentSegmentInfo)
		mock.delInjection(mGetQuerySegmentInfo)
	})
}

func TestReleaseCollection(t *testing.T) {
	ctx := context.Background()

	c := testClient(ctx, t)

	mock.setInjection(mReleaseCollection, func(_ context.Context, raw proto.Message) (proto.Message, error) {
		req, ok := raw.(*server.ReleaseCollectionRequest)
		if !ok {
			return badRequestStatus()
		}
		assert.Equal(t, testCollectionName, req.GetCollectionName())
		return successStatus()
	})

	c.ReleaseCollection(ctx, testCollectionName)
}

func TestGrpcClientHasCollection(t *testing.T) {
	ctx := context.Background()

	c := testClient(ctx, t)

	mock.setInjection(mHasCollection, func(_ context.Context, raw proto.Message) (proto.Message, error) {
		req, ok := raw.(*server.HasCollectionRequest)
		resp := &server.BoolResponse{}
		if !ok {
			s, err := badRequestStatus()
			assert.Fail(t, err.Error())
			resp.Status = s
			return resp, err
		}
		assert.Equal(t, req.CollectionName, testCollectionName)

		s, err := successStatus()
		resp.Status, resp.Value = s, true
		return resp, err
	})

	has, err := c.HasCollection(ctx, testCollectionName)
	assert.Nil(t, err)
	assert.True(t, has)
}

func TestGrpcClientDescribeCollection(t *testing.T) {
	ctx := context.Background()

	c := testClient(ctx, t)

	collectionID := rand.Int63()

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
		resp.CollectionID = collectionID

		s, err := successStatus()
		resp.Status = s

		return resp, err
	})

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

	mock.setInjection(mGetCollectionStatistics, func(_ context.Context, raw proto.Message) (proto.Message, error) {
		req, ok := raw.(*server.GetCollectionStatisticsRequest)
		resp := &server.GetCollectionStatisticsResponse{}
		if !ok {
			s, err := badRequestStatus()
			resp.Status = s
			return resp, err
		}
		assert.Equal(t, testCollectionName, req.GetCollectionName())
		s, err := successStatus()
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

func TestGrpcClientCreatePartition(t *testing.T) {
	//TODO add detail asserts
	ctx := context.Background()
	c := testClient(ctx, t)
	assert.Nil(t, c.CreatePartition(ctx, testCollectionName, "_default_part"))
}

func TestGrpcClientDropPartition(t *testing.T) {
	//TODO add detail asserts
	ctx := context.Background()
	c := testClient(ctx, t)
	assert.Nil(t, c.DropPartition(ctx, testCollectionName, "_default_part"))
}

func TestGrpcClientHasPartition(t *testing.T) {
	//TODO add detail asserts
	ctx := context.Background()
	c := testClient(ctx, t)
	r, err := c.HasPartition(ctx, testCollectionName, "_default_part")
	assert.Nil(t, err)
	assert.False(t, r)
}

// default partition intercetion for ShowPartitions, generates testCollection related paritition data
func getPartitionsInterception(t *testing.T, collName string, partitions ...*entity.Partition) func(ctx context.Context, raw proto.Message) (proto.Message, error) {
	return func(ctx context.Context, raw proto.Message) (proto.Message, error) {
		req, ok := raw.(*server.ShowPartitionsRequest)
		resp := &server.ShowPartitionsResponse{}
		if !ok {
			s, err := badRequestStatus()
			resp.Status = s
			return resp, err
		}
		assert.Equal(t, collName, req.GetCollectionName())
		resp.PartitionIDs = make([]int64, 0, len(partitions))
		resp.PartitionNames = make([]string, 0, len(partitions))
		for _, part := range partitions {
			resp.PartitionIDs = append(resp.PartitionIDs, part.ID)
			resp.PartitionNames = append(resp.PartitionNames, part.Name)
		}
		s, err := successStatus()
		resp.Status = s
		return resp, err
	}
}

func TestGrpcClientShowPartitions(t *testing.T) {

	ctx := context.Background()
	c := testClient(ctx, t)

	type testCase struct {
		collName      string
		partitions    []*entity.Partition
		shouldSuccess bool
	}
	cases := []testCase{
		{
			collName: testCollectionName,
			partitions: []*entity.Partition{
				{
					ID:   1,
					Name: "_part1",
				},
				{
					ID:   2,
					Name: "_part2",
				},
				{
					ID:   3,
					Name: "_part3",
				},
			},
			shouldSuccess: true,
		},
	}
	for _, tc := range cases {
		mock.setInjection(mShowPartitions, getPartitionsInterception(t, tc.collName, tc.partitions...))
		r, err := c.ShowPartitions(ctx, tc.collName)
		if tc.shouldSuccess {
			assert.Nil(t, err)
			assert.NotNil(t, r)
			if assert.Equal(t, len(tc.partitions), len(r)) {
				for idx, part := range tc.partitions {
					assert.Equal(t, part.ID, r[idx].ID)
					assert.Equal(t, part.Name, r[idx].Name)
				}
			}
		} else {
			assert.NotNil(t, err)
		}
	}
}

func TestGrpcClientLoadPartitions(t *testing.T) {
	ctx := context.Background()
	c := testClient(ctx, t)

	type testCase struct {
		collName      string
		partitions    []*entity.Partition
		shouldSuccess bool
		loadNames     []string
	}
	cases := []testCase{
		{
			collName: testCollectionName,
			partitions: []*entity.Partition{
				{
					ID:   1,
					Name: "_part1",
				},
				{
					ID:   2,
					Name: "_part2",
				},
				{
					ID:   3,
					Name: "_part3",
				},
			},
			loadNames:     []string{"_part1", "_part2"},
			shouldSuccess: true,
		},
		{
			collName: testCollectionName,
			partitions: []*entity.Partition{
				{
					ID:   1,
					Name: "_part1",
				},
				{
					ID:   2,
					Name: "_part2",
				},
				{
					ID:   3,
					Name: "_part3",
				},
			},
			loadNames:     []string{"_part4", "_part1"},
			shouldSuccess: false,
		},
	}
	for _, tc := range cases {
		// one segment per paritions
		segmentCount := len(tc.partitions)
		start := time.Now()
		loadTime := rand.Intn(1500) + 100
		rowCounts := rand.Intn(1000) + 100
		ok := false
		mock.setInjection(mGetPersistentSegmentInfo, func(_ context.Context, raw proto.Message) (proto.Message, error) {
			s, err := successStatus()
			r := &server.GetPersistentSegmentInfoResponse{
				Status: s,
				Infos:  make([]*server.PersistentSegmentInfo, 0, segmentCount),
			}
			for i := 0; i < segmentCount; i++ {
				r.Infos = append(r.Infos, &server.PersistentSegmentInfo{
					SegmentID:   int64(i),
					NumRows:     int64(rowCounts),
					PartitionID: tc.partitions[i].ID,
				})
			}
			r.Infos = append(r.Infos, &server.PersistentSegmentInfo{
				SegmentID: int64(segmentCount),
				NumRows:   0, // handcrafted empty segment
			})
			return r, err
		})
		mock.setInjection(mGetQuerySegmentInfo, func(_ context.Context, raw proto.Message) (proto.Message, error) {
			s, err := successStatus()
			r := &server.GetQuerySegmentInfoResponse{
				Status: s,
				Infos:  make([]*server.QuerySegmentInfo, 0, segmentCount),
			}
			rc := 0
			if time.Since(start) > time.Duration(loadTime)*time.Millisecond {
				rc = rowCounts // after load time, row counts set to full amount
				ok = true
			}
			for i := 0; i < segmentCount; i++ {
				r.Infos = append(r.Infos, &server.QuerySegmentInfo{
					SegmentID:   int64(i),
					NumRows:     int64(rc),
					PartitionID: tc.partitions[i].ID,
				})
			}
			return r, err
		})

		mock.setInjection(mShowPartitions, getPartitionsInterception(t, tc.collName, tc.partitions...))
		err := c.LoadPartitions(ctx, tc.collName, tc.loadNames, false)
		if tc.shouldSuccess {
			assert.Nil(t, err)
			assert.True(t, ok)
		} else {
			assert.NotNil(t, err)
		}
	}
}

func TestGrpcClientReleasePartitions(t *testing.T) {
	ctx := context.Background()

	c := testClient(ctx, t)

	parts := []string{"_part1", "_part2"}

	mock.setInjection(mReleasePartitions, func(_ context.Context, raw proto.Message) (proto.Message, error) {
		req, ok := raw.(*server.ReleasePartitionsRequest)
		if !ok {
			return badRequestStatus()
		}
		assert.Equal(t, testCollectionName, req.GetCollectionName())
		assert.ElementsMatch(t, parts, req.GetPartitionNames())

		return successStatus()
	})

	assert.Nil(t, c.ReleasePartitions(ctx, testCollectionName, parts))
}

func TestGrpcClientCreateIndex(t *testing.T) {
	ctx := context.Background()
	c := testClient(ctx, t)
	idx := entity.NewGenericIndex("", entity.Flat, map[string]string{
		"nlist":       "1024",
		"metric_type": "IP",
	})

	assert.Nil(t, c.CreateIndex(ctx, testCollectionName, "vector", idx, true))
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

func successStatus() (*common.Status, error) {
	return &common.Status{ErrorCode: common.ErrorCode_Success}, nil
}

func badRequestStatus() (*common.Status, error) {
	return &common.Status{ErrorCode: common.ErrorCode_IllegalArgument}, errors.New("illegal request type")
}
