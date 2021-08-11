package client

import (
	"context"
	"errors"
	"reflect"
	"testing"

	"github.com/milvus-io/milvus-sdk-go/v2/entity"
	"github.com/milvus-io/milvus-sdk-go/v2/internal/proto/common"
	"github.com/stretchr/testify/assert"
	"google.golang.org/grpc"
)

func TestHandleRespStatus(t *testing.T) {
	assert.NotNil(t, handleRespStatus(nil))
	assert.Nil(t, handleRespStatus(&common.Status{
		ErrorCode: common.ErrorCode_Success,
	}))
	assert.NotNil(t, handleRespStatus(&common.Status{
		ErrorCode: common.ErrorCode_UnexpectedError,
	}))
}

type ValidStruct struct {
	entity.RowBase
	ID     int64 `milvus:"primary_key"`
	Attr1  int8
	Attr2  int16
	Attr3  int32
	Attr4  float32
	Attr5  float64
	Attr6  string
	Vector []float32 `milvus:"dim:128"`
}

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
			m.Name == "CalcDistance" ||
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
				rowType := reflect.TypeOf((*entity.Row)(nil)).Elem()
				switch {
				case inT.Implements(idxType):
					ins = append(ins, reflect.ValueOf(entity.NewFlatIndex("flat_index", entity.L2)))
				case inT.Implements(rowType):
					ins = append(ins, reflect.ValueOf(&ValidStruct{}))
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

func successStatus() (*common.Status, error) {
	return &common.Status{ErrorCode: common.ErrorCode_Success}, nil
}

func badRequestStatus() (*common.Status, error) {
	return &common.Status{ErrorCode: common.ErrorCode_IllegalArgument}, errors.New("illegal request type")
}
