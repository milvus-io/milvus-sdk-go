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
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
	"github.com/milvus-io/milvus-sdk-go/v2/internal/proto/common"
	"github.com/milvus-io/milvus-sdk-go/v2/internal/proto/server"
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
	//	lis.Close()
}

// use bufconn dialer
func bufDialer(context.Context, string) (net.Conn, error) {
	return lis.Dial()
}

func testClient(ctx context.Context, t *testing.T) Client {
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
					entity.TYPE_PARAM_DIM: fmt.Sprintf("%d", testVectorDim),
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

var (
	errNotImplemented = errors.New("not implemented")
)

// type alias for service method
type serviceMethod int

const (
	mCreateCollection        serviceMethod = 1
	mDropCollection          serviceMethod = 2
	mHasCollection           serviceMethod = 3
	mLoadCollection          serviceMethod = 4
	mReleaseCollection       serviceMethod = 5
	mDescribeCollection      serviceMethod = 6
	mListCollection          serviceMethod = 7
	mGetCollectionStatistics serviceMethod = 8
	mShowCollections         serviceMethod = 15
	mCreateAlias             serviceMethod = 16
	mDropAlias               serviceMethod = 17
	mAlterAlias              serviceMethod = 18

	mCreatePartition   serviceMethod = 9
	mDropPartition     serviceMethod = 10
	mHasPartition      serviceMethod = 11
	mLoadPartitions    serviceMethod = 12
	mReleasePartitions serviceMethod = 13
	mShowPartitions    serviceMethod = 14

	mCreateIndex           serviceMethod = 20
	mDropIndex             serviceMethod = 21
	mDescribeIndex         serviceMethod = 22
	mGetIndexState         serviceMethod = 23
	mGetIndexBuildProgress serviceMethod = 24

	mInsert        serviceMethod = 30
	mFlush         serviceMethod = 31
	mSearch        serviceMethod = 32
	mCalcDistance  serviceMethod = 33
	mGetFlushState serviceMethod = 34
	mDelete        serviceMethod = 35
	mQuery         serviceMethod = 36

	mManualCompaction            serviceMethod = 40
	mGetCompactionState          serviceMethod = 41
	mGetCompactionStateWithPlans serviceMethod = 42

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

func (m *mockServer) DescribeCollection(ctx context.Context, req *server.DescribeCollectionRequest) (*server.DescribeCollectionResponse, error) {
	f := m.getInjection(mDescribeCollection)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*server.DescribeCollectionResponse), err
	}
	r := &server.DescribeCollectionResponse{}
	s, err := successStatus()
	r.Status = s
	return r, err
}

func (m *mockServer) GetCollectionStatistics(ctx context.Context, req *server.GetCollectionStatisticsRequest) (*server.GetCollectionStatisticsResponse, error) {
	f := m.getInjection(mGetCollectionStatistics)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*server.GetCollectionStatisticsResponse), err
	}
	r := &server.GetCollectionStatisticsResponse{}
	s, err := successStatus()
	r.Status = s
	return r, err
}

func (m *mockServer) ShowCollections(ctx context.Context, req *server.ShowCollectionsRequest) (*server.ShowCollectionsResponse, error) {
	f := m.getInjection(mShowCollections)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*server.ShowCollectionsResponse), err
	}
	s, err := successStatus()
	return &server.ShowCollectionsResponse{Status: s}, err
}

func (m *mockServer) CreatePartition(ctx context.Context, req *server.CreatePartitionRequest) (*common.Status, error) {
	f := m.getInjection(mCreatePartition)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*common.Status), err
	}
	return successStatus()

}

func (m *mockServer) DropPartition(ctx context.Context, req *server.DropPartitionRequest) (*common.Status, error) {
	f := m.getInjection(mDropPartition)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*common.Status), err
	}
	return successStatus()
}

func (m *mockServer) HasPartition(ctx context.Context, req *server.HasPartitionRequest) (*server.BoolResponse, error) {
	f := m.getInjection(mHasPartition)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*server.BoolResponse), err
	}
	s, err := successStatus()
	return &server.BoolResponse{Status: s, Value: false}, err
}

func (m *mockServer) LoadPartitions(ctx context.Context, req *server.LoadPartitionsRequest) (*common.Status, error) {
	f := m.getInjection(mLoadPartitions)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*common.Status), err
	}
	return successStatus()
}

func (m *mockServer) ReleasePartitions(ctx context.Context, req *server.ReleasePartitionsRequest) (*common.Status, error) {
	f := m.getInjection(mReleasePartitions)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*common.Status), err
	}
	return successStatus()
}

func (m *mockServer) GetPartitionStatistics(_ context.Context, _ *server.GetPartitionStatisticsRequest) (*server.GetPartitionStatisticsResponse, error) {
	panic("not implemented") // TODO: Implement
}

func (m *mockServer) ShowPartitions(ctx context.Context, req *server.ShowPartitionsRequest) (*server.ShowPartitionsResponse, error) {
	f := m.getInjection(mShowPartitions)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*server.ShowPartitionsResponse), err
	}
	s, err := successStatus()
	return &server.ShowPartitionsResponse{Status: s}, err
}

func (m *mockServer) CreateIndex(ctx context.Context, req *server.CreateIndexRequest) (*common.Status, error) {
	f := m.getInjection(mCreateIndex)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*common.Status), err
	}
	return successStatus()
}

func (m *mockServer) DescribeIndex(ctx context.Context, req *server.DescribeIndexRequest) (*server.DescribeIndexResponse, error) {
	f := m.getInjection(mDescribeIndex)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*server.DescribeIndexResponse), err
	}
	s, err := successStatus()
	return &server.DescribeIndexResponse{Status: s}, err
}

func (m *mockServer) GetIndexState(ctx context.Context, req *server.GetIndexStateRequest) (*server.GetIndexStateResponse, error) {
	f := m.getInjection(mGetIndexState)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*server.GetIndexStateResponse), err
	}
	s, err := successStatus()
	return &server.GetIndexStateResponse{Status: s}, err

}

func (m *mockServer) GetIndexBuildProgress(ctx context.Context, req *server.GetIndexBuildProgressRequest) (*server.GetIndexBuildProgressResponse, error) {
	f := m.getInjection(mGetIndexBuildProgress)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*server.GetIndexBuildProgressResponse), err
	}
	s, err := successStatus()
	return &server.GetIndexBuildProgressResponse{Status: s}, err

}

func (m *mockServer) DropIndex(ctx context.Context, req *server.DropIndexRequest) (*common.Status, error) {
	f := m.getInjection(mDropIndex)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*common.Status), err
	}
	return successStatus()
}

func (m *mockServer) Insert(ctx context.Context, req *server.InsertRequest) (*server.MutationResult, error) {
	f := m.getInjection(mInsert)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*server.MutationResult), err
	}
	s, err := successStatus()
	return &server.MutationResult{Status: s}, err
}

func (m *mockServer) Search(ctx context.Context, req *server.SearchRequest) (*server.SearchResults, error) {
	f := m.getInjection(mSearch)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*server.SearchResults), err
	}
	s, err := successStatus()
	return &server.SearchResults{Status: s}, err
}

func (m *mockServer) Flush(ctx context.Context, req *server.FlushRequest) (*server.FlushResponse, error) {
	f := m.getInjection(mFlush)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*server.FlushResponse), err
	}
	s, err := successStatus()
	return &server.FlushResponse{Status: s}, err
}

func (m *mockServer) Query(ctx context.Context, req *server.QueryRequest) (*server.QueryResults, error) {
	f := m.getInjection(mQuery)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*server.QueryResults), err
	}
	s, err := successStatus()
	return &server.QueryResults{Status: s}, err
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

func (m *mockServer) CalcDistance(ctx context.Context, req *server.CalcDistanceRequest) (*server.CalcDistanceResults, error) {
	f := m.getInjection(mCalcDistance)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*server.CalcDistanceResults), err
	}
	resp := &server.CalcDistanceResults{}
	s, err := successStatus()
	resp.Status = s
	return resp, err
}

func (m *mockServer) Delete(ctx context.Context, req *server.DeleteRequest) (*server.MutationResult, error) {
	f := m.getInjection(mDelete)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*server.MutationResult), err
	}
	resp := &server.MutationResult{}
	s, err := successStatus()
	resp.Status = s
	return resp, err
}

// https://wiki.lfaidata.foundation/display/MIL/MEP+8+--+Add+metrics+for+proxy
func (m *mockServer) GetMetrics(_ context.Context, _ *server.GetMetricsRequest) (*server.GetMetricsResponse, error) {
	panic("not implemented") // TODO: Implement
}

func (m *mockServer) CreateAlias(ctx context.Context, req *server.CreateAliasRequest) (*common.Status, error) {
	f := m.getInjection(mCreateAlias)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*common.Status), err
	}
	return successStatus()
}

func (m *mockServer) DropAlias(ctx context.Context, req *server.DropAliasRequest) (*common.Status, error) {
	f := m.getInjection(mDropAlias)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*common.Status), err
	}
	return successStatus()
}

func (m *mockServer) AlterAlias(ctx context.Context, req *server.AlterAliasRequest) (*common.Status, error) {
	f := m.getInjection(mAlterAlias)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*common.Status), err
	}

	return successStatus()
}

func (m *mockServer) GetFlushState(ctx context.Context, req *server.GetFlushStateRequest) (*server.GetFlushStateResponse, error) {
	f := m.getInjection(mGetFlushState)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*server.GetFlushStateResponse), err
	}

	resp := &server.GetFlushStateResponse{}
	s, err := successStatus()
	resp.Status = s
	return resp, err
}

func (m *mockServer) LoadBalance(_ context.Context, _ *server.LoadBalanceRequest) (*common.Status, error) {
	panic("not implemented") // TODO: Implement
}

func (m *mockServer) GetCompactionState(ctx context.Context, req *server.GetCompactionStateRequest) (*server.GetCompactionStateResponse, error) {
	f := m.getInjection(mGetCompactionState)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*server.GetCompactionStateResponse), err
	}

	resp := &server.GetCompactionStateResponse{}
	s, err := successStatus()
	resp.Status = s
	return resp, err
}

func (m *mockServer) ManualCompaction(ctx context.Context, req *server.ManualCompactionRequest) (*server.ManualCompactionResponse, error) {
	f := m.getInjection(mManualCompaction)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*server.ManualCompactionResponse), err
	}
	resp := &server.ManualCompactionResponse{}
	s, err := successStatus()
	resp.Status = s
	return resp, err
}

func (m *mockServer) GetCompactionStateWithPlans(ctx context.Context, req *server.GetCompactionPlansRequest) (*server.GetCompactionPlansResponse, error) {
	f := m.getInjection(mGetCompactionStateWithPlans)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*server.GetCompactionPlansResponse), err
	}

	resp := &server.GetCompactionPlansResponse{}
	s, err := successStatus()
	resp.Status = s
	return resp, err
}
