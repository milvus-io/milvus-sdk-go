package client

import (
	"context"
	"errors"
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
		for j := 1; j < mt.NumIn(); j++ { // idx == 0, is the reciever v
			if j == 1 {
				//non-general solution, hard code context!
				ins = append(ins, reflect.ValueOf(ctx))
				t.Log("skip ctx")
				continue
			}
			inT := mt.In(j)
			t.Log(inT.String())

			switch inT.Kind() {
			case reflect.String: // pass empty
				ins = append(ins, reflect.ValueOf(""))
			case reflect.Int, reflect.Int64:
				ins = append(ins, reflect.ValueOf(0))
			case reflect.Bool:
				ins = append(ins, reflect.ValueOf(false))
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
			outs := v.MethodByName(m.Name).Call(ins)
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
		c := testClient(t, ctx)
		assert.Nil(t, c.Close())
	})

	t.Run("dboule close", func(t *testing.T) {
		c := testClient(t, ctx)
		assert.Nil(t, c.Close())
		assert.Nil(t, c.Close())
	})
}

func TestGrpcClientListCollections(t *testing.T) {
	ctx := context.Background()
	c := testClient(t, ctx)

	c.ListCollections(ctx)
}

func TestGrpcClientCreateCollection(t *testing.T) {
	ctx := context.Background()
	c := testClient(t, ctx)
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
		cases := []entity.Schema{
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
}

func TestGrpcClientDropCollection(t *testing.T) {
	ctx := context.Background()
	c := testClient(t, ctx)

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
	c := testClient(t, ctx)
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

func successStatus() (*common.Status, error) {
	return &common.Status{ErrorCode: common.ErrorCode_Success}, nil
}

func badRequestStatus() (*common.Status, error) {
	return &common.Status{ErrorCode: common.ErrorCode_IllegalArgument}, errors.New("illegal request type")
}
