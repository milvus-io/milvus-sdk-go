package client

import (
	"context"
	"errors"
	"sync"

	"github.com/golang/protobuf/proto"
	common "github.com/milvus-io/milvus-proto/go-api/commonpb"
	server "github.com/milvus-io/milvus-proto/go-api/milvuspb"
)

// ref https://stackoverflow.com/questions/42102496/testing-a-grpc-service

var (
	errNotImplemented = errors.New("not implemented")
)

// type alias for Service method
type ServiceMethod int

const (
	MCreateCollection        ServiceMethod = 101
	MDropCollection          ServiceMethod = 102
	MHasCollection           ServiceMethod = 103
	MLoadCollection          ServiceMethod = 104
	MReleaseCollection       ServiceMethod = 105
	MDescribeCollection      ServiceMethod = 106
	MListCollection          ServiceMethod = 107
	MGetCollectionStatistics ServiceMethod = 108
	MAlterCollection         ServiceMethod = 109
	MGetLoadingProgress      ServiceMethod = 110
	MGetLoadState            ServiceMethod = 111

	MCreatePartition   ServiceMethod = 201
	MDropPartition     ServiceMethod = 202
	MHasPartition      ServiceMethod = 203
	MLoadPartitions    ServiceMethod = 204
	MReleasePartitions ServiceMethod = 205
	MShowPartitions    ServiceMethod = 206

	MShowCollections ServiceMethod = 301
	MCreateAlias     ServiceMethod = 302
	MDropAlias       ServiceMethod = 303
	MAlterAlias      ServiceMethod = 304
	MGetReplicas     ServiceMethod = 305

	MCreateIndex           ServiceMethod = 401
	MDropIndex             ServiceMethod = 402
	MDescribeIndex         ServiceMethod = 403
	MGetIndexState         ServiceMethod = 404
	MGetIndexBuildProgress ServiceMethod = 405

	MCreateCredential ServiceMethod = 500
	MUpdateCredential ServiceMethod = 501
	MDeleteCredential ServiceMethod = 502
	MListCredUsers    ServiceMethod = 503

	MInsert        ServiceMethod = 600
	MFlush         ServiceMethod = 601
	MSearch        ServiceMethod = 602
	MCalcDistance  ServiceMethod = 603
	MGetFlushState ServiceMethod = 604
	MDelete        ServiceMethod = 605
	MQuery         ServiceMethod = 606
	MUpsert        ServiceMethod = 607

	MManualCompaction            ServiceMethod = 700
	MGetCompactionState          ServiceMethod = 701
	MGetCompactionStateWithPlans ServiceMethod = 702

	MGetPersistentSegmentInfo ServiceMethod = 800
	MGetQuerySegmentInfo      ServiceMethod = 801

	MGetComponentStates ServiceMethod = 900
	MGetVersion         ServiceMethod = 901
	MCheckHealth        ServiceMethod = 902
)

// injection function definition
type TestInjection func(context.Context, proto.Message) (proto.Message, error)

// mock Milvus Server
type MockServer struct {
	sync.RWMutex
	Injections map[ServiceMethod]TestInjection
}

func (m *MockServer) SetInjection(n ServiceMethod, f TestInjection) {
	m.Lock()
	defer m.Unlock()
	if m.Injections != nil {
		m.Injections[n] = f
	}
}

func (m *MockServer) GetInjection(n ServiceMethod) TestInjection {
	if m.Injections == nil {
		return nil
	}
	m.RLock()
	defer m.RUnlock()
	return m.Injections[n]
}

func (m *MockServer) DelInjection(n ServiceMethod) {
	if m.Injections == nil {
		return
	}
	m.Lock()
	defer m.Unlock()
	delete(m.Injections, n)
}

func (m *MockServer) CreateCollection(ctx context.Context, req *server.CreateCollectionRequest) (*common.Status, error) {
	f := m.GetInjection(MCreateCollection)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*common.Status), err
	}
	return SuccessStatus()
}

func (m *MockServer) DropCollection(ctx context.Context, req *server.DropCollectionRequest) (*common.Status, error) {
	f := m.GetInjection(MDropCollection)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*common.Status), err
	}

	return SuccessStatus()
}

func (m *MockServer) HasCollection(ctx context.Context, req *server.HasCollectionRequest) (*server.BoolResponse, error) {
	f := m.GetInjection(MHasCollection)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*server.BoolResponse), err
	}
	s, err := SuccessStatus()
	return &server.BoolResponse{Status: s, Value: false}, err
}

func (m *MockServer) LoadCollection(ctx context.Context, req *server.LoadCollectionRequest) (*common.Status, error) {
	f := m.GetInjection(MLoadCollection)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*common.Status), err
	}
	return SuccessStatus()
}

func (m *MockServer) ReleaseCollection(ctx context.Context, req *server.ReleaseCollectionRequest) (*common.Status, error) {
	f := m.GetInjection(MReleaseCollection)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*common.Status), err
	}
	return SuccessStatus()
}

func (m *MockServer) DescribeCollection(ctx context.Context, req *server.DescribeCollectionRequest) (*server.DescribeCollectionResponse, error) {
	f := m.GetInjection(MDescribeCollection)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*server.DescribeCollectionResponse), err
	}
	r := &server.DescribeCollectionResponse{}
	s, err := SuccessStatus()
	r.Status = s
	return r, err
}

func (m *MockServer) GetCollectionStatistics(ctx context.Context, req *server.GetCollectionStatisticsRequest) (*server.GetCollectionStatisticsResponse, error) {
	f := m.GetInjection(MGetCollectionStatistics)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*server.GetCollectionStatisticsResponse), err
	}
	r := &server.GetCollectionStatisticsResponse{}
	s, err := SuccessStatus()
	r.Status = s
	return r, err
}

func (m *MockServer) ShowCollections(ctx context.Context, req *server.ShowCollectionsRequest) (*server.ShowCollectionsResponse, error) {
	f := m.GetInjection(MShowCollections)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*server.ShowCollectionsResponse), err
	}
	s, err := SuccessStatus()
	return &server.ShowCollectionsResponse{Status: s}, err
}

func (m *MockServer) AlterCollection(ctx context.Context, req *server.AlterCollectionRequest) (*common.Status, error) {
	f := m.GetInjection(MAlterCollection)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*common.Status), err
	}
	return SuccessStatus()
}

func (m *MockServer) CreatePartition(ctx context.Context, req *server.CreatePartitionRequest) (*common.Status, error) {
	f := m.GetInjection(MCreatePartition)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*common.Status), err
	}
	return SuccessStatus()

}

func (m *MockServer) DropPartition(ctx context.Context, req *server.DropPartitionRequest) (*common.Status, error) {
	f := m.GetInjection(MDropPartition)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*common.Status), err
	}
	return SuccessStatus()
}

func (m *MockServer) HasPartition(ctx context.Context, req *server.HasPartitionRequest) (*server.BoolResponse, error) {
	f := m.GetInjection(MHasPartition)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*server.BoolResponse), err
	}
	s, err := SuccessStatus()
	return &server.BoolResponse{Status: s, Value: false}, err
}

func (m *MockServer) LoadPartitions(ctx context.Context, req *server.LoadPartitionsRequest) (*common.Status, error) {
	f := m.GetInjection(MLoadPartitions)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*common.Status), err
	}
	return SuccessStatus()
}

func (m *MockServer) ReleasePartitions(ctx context.Context, req *server.ReleasePartitionsRequest) (*common.Status, error) {
	f := m.GetInjection(MReleasePartitions)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*common.Status), err
	}
	return SuccessStatus()
}

func (m *MockServer) GetPartitionStatistics(_ context.Context, _ *server.GetPartitionStatisticsRequest) (*server.GetPartitionStatisticsResponse, error) {
	panic("not implemented") // TODO: Implement
}

func (m *MockServer) ShowPartitions(ctx context.Context, req *server.ShowPartitionsRequest) (*server.ShowPartitionsResponse, error) {
	f := m.GetInjection(MShowPartitions)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*server.ShowPartitionsResponse), err
	}
	s, err := SuccessStatus()
	return &server.ShowPartitionsResponse{Status: s}, err
}

func (m *MockServer) GetLoadingProgress(ctx context.Context, req *server.GetLoadingProgressRequest) (*server.GetLoadingProgressResponse, error) {
	f := m.GetInjection(MGetLoadingProgress)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*server.GetLoadingProgressResponse), err
	}
	s, err := SuccessStatus()
	return &server.GetLoadingProgressResponse{Status: s}, err
}

func (m *MockServer) GetLoadState(ctx context.Context, req *server.GetLoadStateRequest) (*server.GetLoadStateResponse, error) {
	f := m.GetInjection(MGetLoadState)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*server.GetLoadStateResponse), err
	}
	s, err := SuccessStatus()
	return &server.GetLoadStateResponse{Status: s}, err
}

func (m *MockServer) CreateIndex(ctx context.Context, req *server.CreateIndexRequest) (*common.Status, error) {
	f := m.GetInjection(MCreateIndex)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*common.Status), err
	}
	return SuccessStatus()
}

func (m *MockServer) DescribeIndex(ctx context.Context, req *server.DescribeIndexRequest) (*server.DescribeIndexResponse, error) {
	f := m.GetInjection(MDescribeIndex)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*server.DescribeIndexResponse), err
	}
	s, err := SuccessStatus()
	return &server.DescribeIndexResponse{Status: s}, err
}

func (m *MockServer) GetIndexState(ctx context.Context, req *server.GetIndexStateRequest) (*server.GetIndexStateResponse, error) {
	f := m.GetInjection(MGetIndexState)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*server.GetIndexStateResponse), err
	}
	s, err := SuccessStatus()
	return &server.GetIndexStateResponse{Status: s}, err

}

func (m *MockServer) GetIndexBuildProgress(ctx context.Context, req *server.GetIndexBuildProgressRequest) (*server.GetIndexBuildProgressResponse, error) {
	f := m.GetInjection(MGetIndexBuildProgress)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*server.GetIndexBuildProgressResponse), err
	}
	s, err := SuccessStatus()
	return &server.GetIndexBuildProgressResponse{Status: s}, err

}

func (m *MockServer) DropIndex(ctx context.Context, req *server.DropIndexRequest) (*common.Status, error) {
	f := m.GetInjection(MDropIndex)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*common.Status), err
	}
	return SuccessStatus()
}

func (m *MockServer) Insert(ctx context.Context, req *server.InsertRequest) (*server.MutationResult, error) {
	f := m.GetInjection(MInsert)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*server.MutationResult), err
	}
	s, err := SuccessStatus()
	return &server.MutationResult{Status: s}, err
}

func (m *MockServer) Search(ctx context.Context, req *server.SearchRequest) (*server.SearchResults, error) {
	f := m.GetInjection(MSearch)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*server.SearchResults), err
	}
	s, err := SuccessStatus()
	return &server.SearchResults{Status: s}, err
}

func (m *MockServer) Flush(ctx context.Context, req *server.FlushRequest) (*server.FlushResponse, error) {
	f := m.GetInjection(MFlush)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*server.FlushResponse), err
	}
	s, err := SuccessStatus()
	return &server.FlushResponse{Status: s}, err
}

func (m *MockServer) Query(ctx context.Context, req *server.QueryRequest) (*server.QueryResults, error) {
	f := m.GetInjection(MQuery)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*server.QueryResults), err
	}
	s, err := SuccessStatus()
	return &server.QueryResults{Status: s}, err
}

func (m *MockServer) GetPersistentSegmentInfo(ctx context.Context, req *server.GetPersistentSegmentInfoRequest) (*server.GetPersistentSegmentInfoResponse, error) {
	f := m.GetInjection(MGetPersistentSegmentInfo)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*server.GetPersistentSegmentInfoResponse), err
	}
	s, err := SuccessStatus()
	return &server.GetPersistentSegmentInfoResponse{Status: s}, err
}

func (m *MockServer) GetQuerySegmentInfo(ctx context.Context, req *server.GetQuerySegmentInfoRequest) (*server.GetQuerySegmentInfoResponse, error) {
	f := m.GetInjection(MGetQuerySegmentInfo)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*server.GetQuerySegmentInfoResponse), err
	}
	s, err := SuccessStatus()
	return &server.GetQuerySegmentInfoResponse{Status: s}, err
}

func (m *MockServer) Dummy(_ context.Context, _ *server.DummyRequest) (*server.DummyResponse, error) {
	panic("not implemented") // TODO: Implement
}

// TODO: remove
func (m *MockServer) RegisterLink(_ context.Context, _ *server.RegisterLinkRequest) (*server.RegisterLinkResponse, error) {
	panic("not implemented") // TODO: Implement
}

func (m *MockServer) CalcDistance(ctx context.Context, req *server.CalcDistanceRequest) (*server.CalcDistanceResults, error) {
	f := m.GetInjection(MCalcDistance)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*server.CalcDistanceResults), err
	}
	resp := &server.CalcDistanceResults{}
	s, err := SuccessStatus()
	resp.Status = s
	return resp, err
}

func (m *MockServer) Delete(ctx context.Context, req *server.DeleteRequest) (*server.MutationResult, error) {
	f := m.GetInjection(MDelete)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*server.MutationResult), err
	}
	resp := &server.MutationResult{}
	s, err := SuccessStatus()
	resp.Status = s
	return resp, err
}

// https://wiki.lfaidata.foundation/display/MIL/MEP+8+--+Add+metrics+for+proxy
func (m *MockServer) GetMetrics(_ context.Context, _ *server.GetMetricsRequest) (*server.GetMetricsResponse, error) {
	panic("not implemented") // TODO: Implement
}

func (m *MockServer) CreateAlias(ctx context.Context, req *server.CreateAliasRequest) (*common.Status, error) {
	f := m.GetInjection(MCreateAlias)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*common.Status), err
	}
	return SuccessStatus()
}

func (m *MockServer) DropAlias(ctx context.Context, req *server.DropAliasRequest) (*common.Status, error) {
	f := m.GetInjection(MDropAlias)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*common.Status), err
	}
	return SuccessStatus()
}

func (m *MockServer) AlterAlias(ctx context.Context, req *server.AlterAliasRequest) (*common.Status, error) {
	f := m.GetInjection(MAlterAlias)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*common.Status), err
	}

	return SuccessStatus()
}

func (m *MockServer) GetFlushState(ctx context.Context, req *server.GetFlushStateRequest) (*server.GetFlushStateResponse, error) {
	f := m.GetInjection(MGetFlushState)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*server.GetFlushStateResponse), err
	}

	resp := &server.GetFlushStateResponse{}
	s, err := SuccessStatus()
	resp.Status = s
	return resp, err
}

func (m *MockServer) LoadBalance(_ context.Context, _ *server.LoadBalanceRequest) (*common.Status, error) {
	panic("not implemented") // TODO: Implement
}

func (m *MockServer) GetCompactionState(ctx context.Context, req *server.GetCompactionStateRequest) (*server.GetCompactionStateResponse, error) {
	f := m.GetInjection(MGetCompactionState)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*server.GetCompactionStateResponse), err
	}

	resp := &server.GetCompactionStateResponse{}
	s, err := SuccessStatus()
	resp.Status = s
	return resp, err
}

func (m *MockServer) ManualCompaction(ctx context.Context, req *server.ManualCompactionRequest) (*server.ManualCompactionResponse, error) {
	f := m.GetInjection(MManualCompaction)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*server.ManualCompactionResponse), err
	}
	resp := &server.ManualCompactionResponse{}
	s, err := SuccessStatus()
	resp.Status = s
	return resp, err
}

func (m *MockServer) GetCompactionStateWithPlans(ctx context.Context, req *server.GetCompactionPlansRequest) (*server.GetCompactionPlansResponse, error) {
	f := m.GetInjection(MGetCompactionStateWithPlans)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*server.GetCompactionPlansResponse), err
	}

	resp := &server.GetCompactionPlansResponse{}
	s, err := SuccessStatus()
	resp.Status = s
	return resp, err
}

func (m *MockServer) GetReplicas(ctx context.Context, req *server.GetReplicasRequest) (*server.GetReplicasResponse, error) {
	f := m.GetInjection(MGetReplicas)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*server.GetReplicasResponse), err
	}
	resp := &server.GetReplicasResponse{}
	s, err := SuccessStatus()
	resp.Status = s
	return resp, err
}

// https://wiki.lfaidata.foundation/display/MIL/MEP+24+--+Support+bulk+load
func (m *MockServer) Import(_ context.Context, _ *server.ImportRequest) (*server.ImportResponse, error) {
	panic("not implemented") // TODO: Implement
}

func (m *MockServer) GetImportState(_ context.Context, _ *server.GetImportStateRequest) (*server.GetImportStateResponse, error) {
	panic("not implemented") // TODO: Implement
}

func (m *MockServer) ListImportTasks(_ context.Context, _ *server.ListImportTasksRequest) (*server.ListImportTasksResponse, error) {
	panic("not implemented") // TODO: Implement
}

// https://wiki.lfaidata.foundation/display/MIL/MEP+27+--+Support+Basic+Authentication
func (m *MockServer) CreateCredential(ctx context.Context, req *server.CreateCredentialRequest) (*common.Status, error) {
	f := m.GetInjection(MCreateCredential)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*common.Status), err
	}
	return SuccessStatus()
}

func (m *MockServer) UpdateCredential(ctx context.Context, req *server.UpdateCredentialRequest) (*common.Status, error) {
	f := m.GetInjection(MUpdateCredential)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*common.Status), err
	}
	return SuccessStatus()
}

func (m *MockServer) DeleteCredential(ctx context.Context, req *server.DeleteCredentialRequest) (*common.Status, error) {
	f := m.GetInjection(MDeleteCredential)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*common.Status), err
	}
	return SuccessStatus()
}

func (m *MockServer) ListCredUsers(ctx context.Context, req *server.ListCredUsersRequest) (*server.ListCredUsersResponse, error) {
	f := m.GetInjection(MListCredUsers)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*server.ListCredUsersResponse), err
	}
	resp := &server.ListCredUsersResponse{}
	s, err := SuccessStatus()
	resp.Status = s
	return resp, err
}

// https://wiki.lfaidata.foundation/display/MIL/MEP+29+--+Support+Role-Based+Access+Control
func (m *MockServer) CreateRole(_ context.Context, _ *server.CreateRoleRequest) (*common.Status, error) {
	panic("not implemented") // TODO: Implement
}

func (m *MockServer) DropRole(_ context.Context, _ *server.DropRoleRequest) (*common.Status, error) {
	panic("not implemented") // TODO: Implement
}

func (m *MockServer) OperateUserRole(_ context.Context, _ *server.OperateUserRoleRequest) (*common.Status, error) {
	panic("not implemented") // TODO: Implement
}

func (m *MockServer) SelectRole(_ context.Context, _ *server.SelectRoleRequest) (*server.SelectRoleResponse, error) {
	panic("not implemented") // TODO: Implement
}

func (m *MockServer) SelectUser(_ context.Context, _ *server.SelectUserRequest) (*server.SelectUserResponse, error) {
	panic("not implemented") // TODO: Implement
}

func (m *MockServer) OperatePrivilege(_ context.Context, _ *server.OperatePrivilegeRequest) (*common.Status, error) {
	panic("not implemented") // TODO: Implement
}

func (m *MockServer) SelectGrant(_ context.Context, _ *server.SelectGrantRequest) (*server.SelectGrantResponse, error) {
	panic("not implemented") // TODO: Implement
}

//func (m *MockServer) DescribePartition(ctx context.Context, req *server.DescribePartitionRequest) (*server.DescribePartitionResponse, error) {
//	panic("not implemented") // TODO: Implement
//}

func (m *MockServer) GetComponentStates(ctx context.Context, req *server.GetComponentStatesRequest) (*server.ComponentStates, error) {
	f := m.GetInjection(MGetComponentStates)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*server.ComponentStates), err
	}
	s, err := SuccessStatus()
	return &server.ComponentStates{Status: s}, err
}

func (m *MockServer) GetVersion(ctx context.Context, req *server.GetVersionRequest) (*server.GetVersionResponse, error) {
	f := m.GetInjection(MGetVersion)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*server.GetVersionResponse), err
	}
	s, err := SuccessStatus()
	return &server.GetVersionResponse{Status: s}, err
}

func (m *MockServer) CheckHealth(ctx context.Context, req *server.CheckHealthRequest) (*server.CheckHealthResponse, error) {
	f := m.GetInjection(MCheckHealth)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*server.CheckHealthResponse), err
	}
	s, err := SuccessStatus()
	return &server.CheckHealthResponse{Status: s}, err
}

func SuccessStatus() (*common.Status, error) {
	return &common.Status{ErrorCode: common.ErrorCode_Success}, nil
}

func BadRequestStatus() (*common.Status, error) {
	return &common.Status{ErrorCode: common.ErrorCode_IllegalArgument}, errors.New("illegal request type")
}

func BadStatus() (*common.Status, error) {
	return &common.Status{
		ErrorCode: common.ErrorCode_UnexpectedError,
		Reason:    "fail reason",
	}, nil
}

func (m *MockServer) Upsert(ctx context.Context, req *server.UpsertRequest) (*server.MutationResult, error) {
	f := m.GetInjection(MUpsert)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*server.MutationResult), err
	}
	s, err := SuccessStatus()
	return &server.MutationResult{Status: s}, err
}
