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
	common "github.com/milvus-io/milvus-proto/go-api/commonpb"
	server "github.com/milvus-io/milvus-proto/go-api/milvuspb"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
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

var (
	errNotImplemented = errors.New("not implemented")
)

// type alias for service method
type serviceMethod int

const (
	mCreateCollection        serviceMethod = 101
	mDropCollection          serviceMethod = 102
	mHasCollection           serviceMethod = 103
	mLoadCollection          serviceMethod = 104
	mReleaseCollection       serviceMethod = 105
	mDescribeCollection      serviceMethod = 106
	mListCollection          serviceMethod = 107
	mGetCollectionStatistics serviceMethod = 108
	mAlterCollection         serviceMethod = 109
	mGetLoadingProgress      serviceMethod = 110

	mCreatePartition   serviceMethod = 201
	mDropPartition     serviceMethod = 202
	mHasPartition      serviceMethod = 203
	mLoadPartitions    serviceMethod = 204
	mReleasePartitions serviceMethod = 205
	mShowPartitions    serviceMethod = 206

	mShowCollections serviceMethod = 301
	mCreateAlias     serviceMethod = 302
	mDropAlias       serviceMethod = 303
	mAlterAlias      serviceMethod = 304
	mGetReplicas     serviceMethod = 305

	mCreateIndex           serviceMethod = 401
	mDropIndex             serviceMethod = 402
	mDescribeIndex         serviceMethod = 403
	mGetIndexState         serviceMethod = 404
	mGetIndexBuildProgress serviceMethod = 405

	mCreateCredential serviceMethod = 500
	mUpdateCredential serviceMethod = 501
	mDeleteCredential serviceMethod = 502
	mListCredUsers    serviceMethod = 503

	mInsert        serviceMethod = 600
	mFlush         serviceMethod = 601
	mSearch        serviceMethod = 602
	mCalcDistance  serviceMethod = 603
	mGetFlushState serviceMethod = 604
	mDelete        serviceMethod = 605
	mQuery         serviceMethod = 606

	mManualCompaction            serviceMethod = 700
	mGetCompactionState          serviceMethod = 701
	mGetCompactionStateWithPlans serviceMethod = 702

	mGetPersistentSegmentInfo serviceMethod = 800
	mGetQuerySegmentInfo      serviceMethod = 801

	mGetComponentStates serviceMethod = 900
	mGetVersion         serviceMethod = 901
	mCheckHealth        serviceMethod = 902
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

func (m *mockServer) AlterCollection(ctx context.Context, req *server.AlterCollectionRequest) (*common.Status, error) {
	f := m.getInjection(mAlterCollection)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*common.Status), err
	}
	return successStatus()
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

func (m *mockServer) GetLoadingProgress(ctx context.Context, req *server.GetLoadingProgressRequest) (*server.GetLoadingProgressResponse, error) {
	f := m.getInjection(mGetLoadingProgress)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*server.GetLoadingProgressResponse), err
	}
	s, err := successStatus()
	return &server.GetLoadingProgressResponse{Status: s}, err
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

func (m *mockServer) GetReplicas(ctx context.Context, req *server.GetReplicasRequest) (*server.GetReplicasResponse, error) {
	f := m.getInjection(mGetReplicas)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*server.GetReplicasResponse), err
	}
	resp := &server.GetReplicasResponse{}
	s, err := successStatus()
	resp.Status = s
	return resp, err
}

// https://wiki.lfaidata.foundation/display/MIL/MEP+24+--+Support+bulk+load
func (m *mockServer) Import(_ context.Context, _ *server.ImportRequest) (*server.ImportResponse, error) {
	panic("not implemented") // TODO: Implement
}

func (m *mockServer) GetImportState(_ context.Context, _ *server.GetImportStateRequest) (*server.GetImportStateResponse, error) {
	panic("not implemented") // TODO: Implement
}

func (m *mockServer) ListImportTasks(_ context.Context, _ *server.ListImportTasksRequest) (*server.ListImportTasksResponse, error) {
	panic("not implemented") // TODO: Implement
}

// https://wiki.lfaidata.foundation/display/MIL/MEP+27+--+Support+Basic+Authentication
func (m *mockServer) CreateCredential(ctx context.Context, req *server.CreateCredentialRequest) (*common.Status, error) {
	f := m.getInjection(mCreateCredential)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*common.Status), err
	}
	return successStatus()
}

func (m *mockServer) UpdateCredential(ctx context.Context, req *server.UpdateCredentialRequest) (*common.Status, error) {
	f := m.getInjection(mUpdateCredential)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*common.Status), err
	}
	return successStatus()
}

func (m *mockServer) DeleteCredential(ctx context.Context, req *server.DeleteCredentialRequest) (*common.Status, error) {
	f := m.getInjection(mDeleteCredential)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*common.Status), err
	}
	return successStatus()
}

func (m *mockServer) ListCredUsers(ctx context.Context, req *server.ListCredUsersRequest) (*server.ListCredUsersResponse, error) {
	f := m.getInjection(mListCredUsers)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*server.ListCredUsersResponse), err
	}
	resp := &server.ListCredUsersResponse{}
	s, err := successStatus()
	resp.Status = s
	return resp, err
}

// https://wiki.lfaidata.foundation/display/MIL/MEP+29+--+Support+Role-Based+Access+Control
func (m *mockServer) CreateRole(_ context.Context, _ *server.CreateRoleRequest) (*common.Status, error) {
	panic("not implemented") // TODO: Implement
}

func (m *mockServer) DropRole(_ context.Context, _ *server.DropRoleRequest) (*common.Status, error) {
	panic("not implemented") // TODO: Implement
}

func (m *mockServer) OperateUserRole(_ context.Context, _ *server.OperateUserRoleRequest) (*common.Status, error) {
	panic("not implemented") // TODO: Implement
}

func (m *mockServer) SelectRole(_ context.Context, _ *server.SelectRoleRequest) (*server.SelectRoleResponse, error) {
	panic("not implemented") // TODO: Implement
}

func (m *mockServer) SelectUser(_ context.Context, _ *server.SelectUserRequest) (*server.SelectUserResponse, error) {
	panic("not implemented") // TODO: Implement
}

func (m *mockServer) OperatePrivilege(_ context.Context, _ *server.OperatePrivilegeRequest) (*common.Status, error) {
	panic("not implemented") // TODO: Implement
}

func (m *mockServer) SelectGrant(_ context.Context, _ *server.SelectGrantRequest) (*server.SelectGrantResponse, error) {
	panic("not implemented") // TODO: Implement
}

//func (m *mockServer) DescribePartition(ctx context.Context, req *server.DescribePartitionRequest) (*server.DescribePartitionResponse, error) {
//	panic("not implemented") // TODO: Implement
//}

func (m *mockServer) GetComponentStates(ctx context.Context, req *server.GetComponentStatesRequest) (*server.ComponentStates, error) {
	f := m.getInjection(mGetComponentStates)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*server.ComponentStates), err
	}
	s, err := successStatus()
	return &server.ComponentStates{Status: s}, err
}

func (m *mockServer) GetVersion(ctx context.Context, req *server.GetVersionRequest) (*server.GetVersionResponse, error) {
	f := m.getInjection(mGetVersion)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*server.GetVersionResponse), err
	}
	s, err := successStatus()
	return &server.GetVersionResponse{Status: s}, err
}

func (m *mockServer) CheckHealth(ctx context.Context, req *server.CheckHealthRequest) (*server.CheckHealthResponse, error) {
	f := m.getInjection(mCheckHealth)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*server.CheckHealthResponse), err
	}
	s, err := successStatus()
	return &server.CheckHealthResponse{Status: s}, err
}
