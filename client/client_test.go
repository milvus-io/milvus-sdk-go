package client

import (
	"context"
	"errors"
	"fmt"
	"log"
	"net"
	"reflect"
	"strings"
	"testing"
	"time"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/examples/helloworld/helloworld"
	"google.golang.org/grpc/keepalive"
	"google.golang.org/grpc/reflection"
	"google.golang.org/grpc/status"

	common "github.com/milvus-io/milvus-proto/go-api/commonpb"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
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
		t.Run(fmt.Sprintf("TestGrpcClientNil_%s", m.Name), func(t *testing.T) {
			mt := m.Type                                   // type of function
			if m.Name == "Close" || m.Name == "Connect" || // skip connect & close
				m.Name == "Search" || // type alias MetricType treated as string
				m.Name == "CalcDistance" ||
				m.Name == "ManualCompaction" || // time.Duration hard to detect in reflect
				m.Name == "Insert" { // complex methods with ...
				t.Skip("method", m.Name, "skipped")
			}
			ins := make([]reflect.Value, 0, mt.NumIn())
			for j := 1; j < mt.NumIn(); j++ { // idx == 0, is the receiver v
				if j == 1 {
					//non-general solution, hard code context!
					ins = append(ins, reflect.ValueOf(ctx))
					continue
				}
				if mt.IsVariadic() {
					// Variadic function, skip last parameter
					// func m (arg1 interface, opts ... options)
					if j == mt.NumIn()-1 {
						continue
					}
				}
				inT := mt.In(j)

				switch inT.Kind() {
				case reflect.String: // pass empty
					ins = append(ins, reflect.ValueOf(""))
				case reflect.Int:
					ins = append(ins, reflect.ValueOf(0))
				case reflect.Int64:
					ins = append(ins, reflect.ValueOf(int64(0)))
				case reflect.Bool:
					ins = append(ins, reflect.ValueOf(false))
				case reflect.Interface:
					idxType := reflect.TypeOf((*entity.Index)(nil)).Elem()
					rowType := reflect.TypeOf((*entity.Row)(nil)).Elem()
					colType := reflect.TypeOf((*entity.Column)(nil)).Elem()
					switch {
					case inT.Implements(idxType):
						ins = append(ins, reflect.ValueOf(entity.NewFlatIndex("flat_index", entity.L2)))
					case inT.Implements(rowType):
						ins = append(ins, reflect.ValueOf(&ValidStruct{}))
					case inT.Implements(colType):
						ins = append(ins, reflect.ValueOf(entity.NewColumnInt64("id", []int64{})))
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

		})
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

type Tserver struct {
	helloworld.UnimplementedGreeterServer
	reqCounter   uint
	SuccessCount uint
}

func (s *Tserver) SayHello(ctx context.Context, in *helloworld.HelloRequest) (*helloworld.HelloReply, error) {
	log.Printf("Received: %s", in.Name)
	s.reqCounter++
	if s.reqCounter%s.SuccessCount == 0 {
		log.Printf("success %d", s.reqCounter)
		return &helloworld.HelloReply{Message: strings.ToUpper(in.Name)}, nil
	}
	return nil, status.Errorf(codes.Unavailable, "server: fail it")
}

func TestGrpcClientRetryPolicy(t *testing.T) {
	// server
	port := ":50051"
	address := "localhost:50051"
	lis, err := net.Listen("tcp", port)
	if err != nil {
		log.Fatalf("failed to listen: %v", err)
	}
	var kaep = keepalive.EnforcementPolicy{
		MinTime:             5 * time.Second,
		PermitWithoutStream: true,
	}
	var kasp = keepalive.ServerParameters{
		Time:    60 * time.Second,
		Timeout: 60 * time.Second,
	}

	maxAttempts := 5
	s := grpc.NewServer(
		grpc.KeepaliveEnforcementPolicy(kaep),
		grpc.KeepaliveParams(kasp),
	)
	helloworld.RegisterGreeterServer(s, &Tserver{SuccessCount: uint(maxAttempts)})
	reflection.Register(s)
	go func() {
		if err := s.Serve(lis); err != nil {
			log.Fatalf("failed to serve: %v", err)
		}
	}()
	defer s.Stop()

	client, err := NewGrpcClient(context.TODO(), address)
	assert.Nil(t, err)
	defer client.Close()

	greeterClient := helloworld.NewGreeterClient(client.(*grpcClient).conn)
	ctx := context.Background()
	name := fmt.Sprintf("hello world %d", time.Now().Second())
	res, err := greeterClient.SayHello(ctx, &helloworld.HelloRequest{Name: name})
	assert.Nil(t, err)
	assert.Equal(t, res.Message, strings.ToUpper(name))
}

func TestClient_getDefaultAuthOpts(t *testing.T) {
	username := "u"
	password := "pwd"
	defaultOpts := getDefaultAuthOpts(username, password, true)
	assert.True(t, len(defaultOpts) == 6)

	defaultOpts = getDefaultAuthOpts(username, password, false)
	assert.True(t, len(defaultOpts) == 6)
}

func successStatus() (*common.Status, error) {
	return &common.Status{ErrorCode: common.ErrorCode_Success}, nil
}

func badRequestStatus() (*common.Status, error) {
	return &common.Status{ErrorCode: common.ErrorCode_IllegalArgument}, errors.New("illegal request type")
}
