package client

import (
	"context"
	"fmt"
	"log"
	"math/rand"
	"net"
	"reflect"
	"strings"
	"testing"
	"time"

	common "github.com/milvus-io/milvus-proto/go-api/commonpb"
	server "github.com/milvus-io/milvus-proto/go-api/milvuspb"
	"github.com/stretchr/testify/assert"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/examples/helloworld/helloworld"
	"google.golang.org/grpc/keepalive"
	"google.golang.org/grpc/reflection"
	"google.golang.org/grpc/status"
	"google.golang.org/grpc/test/bufconn"

	"github.com/milvus-io/milvus-sdk-go/v2/entity"
)

const (
	bufSize = 1024 * 1024
)

var (
	lis        *bufconn.Listener
	mockServer *MockServer
)

const (
	testCollectionName       = `test_go_sdk`
	testCollectionID         = int64(789)
	testPrimaryField         = `int64`
	testVectorField          = `vector`
	testVectorDim            = 128
	testDefaultReplicaNumber = int32(1)
	testMultiReplicaNumber   = int32(2)
	testUsername             = "user"
	testPassword             = "pwd"
)

func defaultSchema() *entity.Schema {
	return &entity.Schema{
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
				Name:     testVectorField,
				DataType: entity.FieldTypeFloatVector,
				TypeParams: map[string]string{
					entity.TypeParamDim: fmt.Sprintf("%d", testVectorDim),
				},
			},
		},
	}
}

func varCharSchema() *entity.Schema {
	return &entity.Schema{
		CollectionName: testCollectionName,
		AutoID:         false,
		Fields: []*entity.Field{
			{
				Name:       "varchar",
				DataType:   entity.FieldTypeVarChar,
				PrimaryKey: true,
				AutoID:     false,
				TypeParams: map[string]string{
					entity.TypeParamMaxLength: fmt.Sprintf("%d", 100),
				},
			},
			{
				Name:     testVectorField,
				DataType: entity.FieldTypeFloatVector,
				TypeParams: map[string]string{
					entity.TypeParamDim: fmt.Sprintf("%d", testVectorDim),
				},
			},
		},
	}
}

var _ entity.Row = &defaultRow{}

type defaultRow struct {
	entity.RowBase
	int64  int64     `milvus:"primary_key"`
	Vector []float32 `milvus:"dim:128"`
}

func (r defaultRow) Collection() string {
	return testCollectionName
}

// TestMain establishes mock grpc server to testing client behavior
func TestMain(m *testing.M) {
	rand.Seed(time.Now().Unix())
	lis = bufconn.Listen(bufSize)
	s := grpc.NewServer()
	mockServer = &MockServer{
		Injections: make(map[ServiceMethod]TestInjection),
	}
	server.RegisterMilvusServiceServer(s, mockServer)
	go func() {
		if err := s.Serve(lis); err != nil {
			log.Fatalf("Server exited with error: %v", err)
		}
	}()
	m.Run()
	//	lis.Close()
}

// use bufconn dialer
func bufDialer(context.Context, string) (net.Conn, error) {
	return lis.Dial()
}

func testClient(ctx context.Context, t *testing.T) Client {
	c, err := NewClient(ctx,
		Config{
			Address: "bufnet",
			DialOptions: []grpc.DialOption{
				grpc.WithBlock(),
				grpc.WithInsecure(),
				grpc.WithContextDialer(bufDialer),
			},
		})

	if !assert.Nil(t, err) || !assert.NotNil(t, c) {
		t.FailNow()
	}
	return c
}

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
	c := &GrpcClient{}
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
				m.Name == "UsingDatabase" || // skip use database
				m.Name == "Search" || // type alias MetricType treated as string
				m.Name == "CalcDistance" ||
				m.Name == "ManualCompaction" || // time.Duration hard to detect in reflect
				m.Name == "Insert" { // complex methods with ...
				t.Skip("method", m.Name, "skipped")
			}
			ins := make([]reflect.Value, 0, mt.NumIn())
			for j := 1; j < mt.NumIn(); j++ { // idx == 0, is the receiver v
				if j == 1 {
					// non-general solution, hard code context!
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
						idx, _ := entity.NewIndexFlat(entity.L2)
						ins = append(ins, reflect.ValueOf(idx))
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
		c, err := NewClient(ctx,
			Config{
				Address: "bufnet",
				DialOptions: []grpc.DialOption{
					grpc.WithBlock(), grpc.WithInsecure(), grpc.WithContextDialer(bufDialer),
				},
			})
		assert.Nil(t, err)
		assert.NotNil(t, c)
	})

	t.Run("Test empty addr, using default timeout", func(t *testing.T) {
		c, err := NewClient(ctx, Config{
			Address: "",
		})
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
	kaep := keepalive.EnforcementPolicy{
		MinTime:             5 * time.Second,
		PermitWithoutStream: true,
	}
	kasp := keepalive.ServerParameters{
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

	client, err := NewClient(context.TODO(), Config{Address: address, DisableConn: true})
	assert.Nil(t, err)
	defer client.Close()

	greeterClient := helloworld.NewGreeterClient(client.(*GrpcClient).Conn)
	ctx := context.Background()
	name := fmt.Sprintf("hello world %d", time.Now().Second())
	res, err := greeterClient.SayHello(ctx, &helloworld.HelloRequest{Name: name})
	assert.Nil(t, err)
	assert.Equal(t, res.Message, strings.ToUpper(name))
}

func TestClient_NewDefaultGrpcClientWithURI(t *testing.T) {
	username := "u"
	password := "p"
	t.Run("create grpc client fail", func(t *testing.T) {
		uri := "https://localhost:port"
		ctx := context.Background()
		client, err := NewDefaultGrpcClientWithURI(ctx, uri, username, password)
		assert.Nil(t, client)
		assert.Error(t, err)
	})
}

var letters = []rune("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")

func randStr(n int) string {
	sb := strings.Builder{}
	sb.Grow(n)

	for i := 0; i < n; i++ {
		sb.WriteRune(letters[rand.Intn(len(letters))])
	}

	return sb.String()
}
