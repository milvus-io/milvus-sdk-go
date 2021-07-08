package client

import (
	"context"
	"errors"
	"fmt"
	"log"
	"math/rand"
	"net"
	"sync"
	"testing"
	"time"

	"github.com/golang/protobuf/proto"
	"github.com/milvus-io/milvus-sdk-go/entity"
	"github.com/milvus-io/milvus-sdk-go/internal/proto/common"
	"github.com/milvus-io/milvus-sdk-go/internal/proto/server"
	"github.com/stretchr/testify/assert"
	"google.golang.org/grpc"
	"google.golang.org/grpc/test/bufconn"
)

// ref https://stackoverflow.com/questions/42102496/testing-a-grpc-service

const (
	bufSzie = 1024 * 1024
)

var (
	lis  *bufconn.Listener
	mock *mockServer
)

// TestMain establishes mock grpc server to testing client behavior
func TestMain(m *testing.M) {
	rand.Seed(time.Now().Unix())
	lis = bufconn.Listen(bufSzie)
	s := grpc.NewServer()
	mock = &mockServer{
		injections: make(map[serviceMethod]testInjection),
	}
	server.RegisterMilvusServiceServer(s, mock)
	go func() {
		if err := s.Serve(lis); err != nil {
			log.Fatalf("Server exited with error: %v", err)
		}
	}()
	m.Run()
	lis.Close()
}

// use bufconn dialer
func bufDialer(context.Context, string) (net.Conn, error) {
	return lis.Dial()
}

func testClient(t *testing.T, ctx context.Context) Client {
	c, err := NewGrpcClient(ctx, "bufnet", grpc.WithBlock(),
		grpc.WithInsecure(), grpc.WithContextDialer(bufDialer))

	if !assert.Nil(t, err) || !assert.NotNil(t, c) {
		t.FailNow()
	}
	return c
}

const (
	testCollectionName = `test_go_sdk`
	testPrimaryField   = `int64`
	testVectorField    = `vector`
	testVectorDim      = 128
)

func defaultSchema() entity.Schema {
	return entity.Schema{
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
					"dim": fmt.Sprintf("%d", testVectorDim),
				},
			},
		},
	}
}

var (
	errNotImplemented = errors.New("not implemented")
)

// type alias for service method
type serviceMethod int

const (
	mCreateCollection   serviceMethod = 1
	mDropCollection     serviceMethod = 2
	mHasCollection      serviceMethod = 3
	mLoadCollection     serviceMethod = 4
	mReleaseCollection  serviceMethod = 5
	mDescribeCollection serviceMethod = 6
	mListCollection     serviceMethod = 7

	mGetPersistentSegmentInfo serviceMethod = 98
	mGetQuerySegmentInfo      serviceMethod = 99
)

// injection function definition
type testInjection func(context.Context, proto.Message) (proto.Message, error)

// mock Milvus Server
type mockServer struct {
	sync.RWMutex
	injections map[serviceMethod]testInjection
}

func (m *mockServer) setInjection(n serviceMethod, f testInjection) {
	m.Lock()
	defer m.Unlock()
	if m.injections != nil {
		m.injections[n] = f
	}
}

func (m *mockServer) getInjection(n serviceMethod) testInjection {
	if m.injections == nil {
		return nil
	}
	m.RLock()
	defer m.RUnlock()
	return m.injections[n]
}

func (m *mockServer) delInjection(n serviceMethod) {
	if m.injections == nil {
		return
	}
	m.Lock()
	defer m.Unlock()
	delete(m.injections, n)
}

func (m *mockServer) CreateCollection(ctx context.Context, req *server.CreateCollectionRequest) (*common.Status, error) {
	f := m.getInjection(mCreateCollection)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*common.Status), err
	}
	return successStatus()
}

func (m *mockServer) DropCollection(ctx context.Context, req *server.DropCollectionRequest) (*common.Status, error) {
	f := m.getInjection(mDropCollection)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*common.Status), err
	}

	return successStatus()
}

func (m *mockServer) HasCollection(ctx context.Context, req *server.HasCollectionRequest) (*server.BoolResponse, error) {
	f := m.getInjection(mHasCollection)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*server.BoolResponse), err
	}
	s, err := successStatus()
	return &server.BoolResponse{Status: s, Value: false}, err
}

func (m *mockServer) LoadCollection(ctx context.Context, req *server.LoadCollectionRequest) (*common.Status, error) {
	f := m.getInjection(mLoadCollection)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*common.Status), err
	}
	return successStatus()
}

func (m *mockServer) ReleaseCollection(ctx context.Context, req *server.ReleaseCollectionRequest) (*common.Status, error) {
	f := m.getInjection(mReleaseCollection)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*common.Status), err
	}
	return successStatus()

}

func (m *mockServer) DescribeCollection(_ context.Context, _ *server.DescribeCollectionRequest) (*server.DescribeCollectionResponse, error) {
	panic("not implemented") // TODO: Implement
}

func (m *mockServer) GetCollectionStatistics(_ context.Context, _ *server.GetCollectionStatisticsRequest) (*server.GetCollectionStatisticsResponse, error) {
	panic("not implemented") // TODO: Implement
}

func (m *mockServer) ShowCollections(ctx context.Context, req *server.ShowCollectionsRequest) (*server.ShowCollectionsResponse, error) {
	f := m.getInjection(mListCollection)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*server.ShowCollectionsResponse), err
	}
	s, err := successStatus()
	return &server.ShowCollectionsResponse{Status: s}, err
}

func (m *mockServer) CreatePartition(_ context.Context, _ *server.CreatePartitionRequest) (*common.Status, error) {
	panic("not implemented") // TODO: Implement
}

func (m *mockServer) DropPartition(_ context.Context, _ *server.DropPartitionRequest) (*common.Status, error) {
	panic("not implemented") // TODO: Implement
}

func (m *mockServer) HasPartition(_ context.Context, _ *server.HasPartitionRequest) (*server.BoolResponse, error) {
	panic("not implemented") // TODO: Implement
}

func (m *mockServer) LoadPartitions(_ context.Context, _ *server.LoadPartitionsRequest) (*common.Status, error) {
	panic("not implemented") // TODO: Implement
}

func (m *mockServer) ReleasePartitions(_ context.Context, _ *server.ReleasePartitionsRequest) (*common.Status, error) {
	panic("not implemented") // TODO: Implement
}

func (m *mockServer) GetPartitionStatistics(_ context.Context, _ *server.GetPartitionStatisticsRequest) (*server.GetPartitionStatisticsResponse, error) {
	panic("not implemented") // TODO: Implement
}

func (m *mockServer) ShowPartitions(_ context.Context, _ *server.ShowPartitionsRequest) (*server.ShowPartitionsResponse, error) {
	panic("not implemented") // TODO: Implement
}

func (m *mockServer) CreateIndex(_ context.Context, _ *server.CreateIndexRequest) (*common.Status, error) {
	panic("not implemented") // TODO: Implement
}

func (m *mockServer) DescribeIndex(_ context.Context, _ *server.DescribeIndexRequest) (*server.DescribeIndexResponse, error) {
	panic("not implemented") // TODO: Implement
}

func (m *mockServer) GetIndexState(_ context.Context, _ *server.GetIndexStateRequest) (*server.GetIndexStateResponse, error) {
	panic("not implemented") // TODO: Implement
}

func (m *mockServer) GetIndexBuildProgress(_ context.Context, _ *server.GetIndexBuildProgressRequest) (*server.GetIndexBuildProgressResponse, error) {
	panic("not implemented") // TODO: Implement
}

func (m *mockServer) DropIndex(_ context.Context, _ *server.DropIndexRequest) (*common.Status, error) {
	panic("not implemented") // TODO: Implement
}

func (m *mockServer) Insert(_ context.Context, _ *server.InsertRequest) (*server.MutationResult, error) {
	panic("not implemented") // TODO: Implement
}

func (m *mockServer) Search(_ context.Context, _ *server.SearchRequest) (*server.SearchResults, error) {
	panic("not implemented") // TODO: Implement
}

func (m *mockServer) Retrieve(_ context.Context, _ *server.RetrieveRequest) (*server.RetrieveResults, error) {
	panic("not implemented") // TODO: Implement
}

func (m *mockServer) Flush(_ context.Context, _ *server.FlushRequest) (*server.FlushResponse, error) {
	panic("not implemented") // TODO: Implement
}

func (m *mockServer) Query(_ context.Context, _ *server.QueryRequest) (*server.QueryResults, error) {
	panic("not implemented") // TODO: Implement
}

func (m *mockServer) GetPersistentSegmentInfo(ctx context.Context, req *server.GetPersistentSegmentInfoRequest) (*server.GetPersistentSegmentInfoResponse, error) {
	f := m.getInjection(mGetPersistentSegmentInfo)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*server.GetPersistentSegmentInfoResponse), err
	}
	s, err := successStatus()
	return &server.GetPersistentSegmentInfoResponse{Status: s}, err
}

func (m *mockServer) GetQuerySegmentInfo(ctx context.Context, req *server.GetQuerySegmentInfoRequest) (*server.GetQuerySegmentInfoResponse, error) {
	f := m.getInjection(mGetQuerySegmentInfo)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*server.GetQuerySegmentInfoResponse), err
	}
	s, err := successStatus()
	return &server.GetQuerySegmentInfoResponse{Status: s}, err
}

func (m *mockServer) Dummy(_ context.Context, _ *server.DummyRequest) (*server.DummyResponse, error) {
	panic("not implemented") // TODO: Implement
}

// TODO: remove
func (m *mockServer) RegisterLink(_ context.Context, _ *server.RegisterLinkRequest) (*server.RegisterLinkResponse, error) {
	panic("not implemented") // TODO: Implement
}
