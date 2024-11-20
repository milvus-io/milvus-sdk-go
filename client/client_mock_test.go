package client

import (
	"context"
	"net"
	"sync"

	"github.com/cockroachdb/errors"

	"github.com/golang/protobuf/proto"
	"github.com/milvus-io/milvus-proto/go-api/v2/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v2/federpb"
	"github.com/milvus-io/milvus-proto/go-api/v2/milvuspb"
	"github.com/milvus-io/milvus-proto/go-api/v2/schemapb"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
	"github.com/milvus-io/milvus-sdk-go/v2/merr"
	"github.com/milvus-io/milvus-sdk-go/v2/mocks"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/suite"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/test/bufconn"
)

type MockSuiteBase struct {
	suite.Suite

	lis  *bufconn.Listener
	svr  *grpc.Server
	mock *mocks.MilvusServiceServer

	client *GrpcClient
}

func (s *MockSuiteBase) SetupSuite() {
	s.lis = bufconn.Listen(bufSize)
	s.svr = grpc.NewServer()

	s.mock = &mocks.MilvusServiceServer{}

	milvuspb.RegisterMilvusServiceServer(s.svr, s.mock)

	go func() {
		s.T().Log("start mock server")
		if err := s.svr.Serve(s.lis); err != nil {
			s.Fail("failed to start mock server", err.Error())
		}
	}()
	s.setupConnect()
}

func (s *MockSuiteBase) TearDownSuite() {
	s.svr.Stop()
	s.lis.Close()
}

func (s *MockSuiteBase) mockDialer(context.Context, string) (net.Conn, error) {
	return s.lis.Dial()
}

func (s *MockSuiteBase) SetupTest() {
	c, err := NewClient(context.Background(), Config{
		Address:     "bufnet2",
		DisableConn: true,
		DialOptions: []grpc.DialOption{
			grpc.WithBlock(),
			grpc.WithTransportCredentials(insecure.NewCredentials()),
			grpc.WithContextDialer(s.mockDialer),
		},
	})
	s.Require().NoError(err)
	s.setupConnect()

	grpcClient, ok := c.(*GrpcClient)
	s.Require().True(ok)
	s.client = grpcClient
}

func (s *MockSuiteBase) TearDownTest() {
	s.client.Close()
	s.client = nil
}

func (s *MockSuiteBase) resetMock() {
	// MetaCache.reset()
	if s.mock != nil {
		s.client.cache.reset()
		s.mock.Calls = nil
		s.mock.ExpectedCalls = nil
		s.setupConnect()
	}
}

func (s *MockSuiteBase) setupConnect() {
	s.mock.EXPECT().Connect(mock.Anything, mock.AnythingOfType("*milvuspb.ConnectRequest")).
		Return(&milvuspb.ConnectResponse{
			Status:     &commonpb.Status{},
			Identifier: 1,
		}, nil).Maybe()
}

func (s *MockSuiteBase) setupHasCollection(collNames ...string) {
	s.mock.EXPECT().HasCollection(mock.Anything, mock.AnythingOfType("*milvuspb.HasCollectionRequest")).
		Call.Return(func(ctx context.Context, req *milvuspb.HasCollectionRequest) *milvuspb.BoolResponse {
		resp := &milvuspb.BoolResponse{Status: &commonpb.Status{}}
		for _, collName := range collNames {
			if req.GetCollectionName() == collName {
				resp.Value = true
				break
			}
		}
		return resp
	}, nil)
}

func (s *MockSuiteBase) setupHasCollectionError(errorCode commonpb.ErrorCode, err error) {
	s.mock.EXPECT().HasCollection(mock.Anything, mock.AnythingOfType("*milvuspb.HasCollectionRequest")).
		Return(&milvuspb.BoolResponse{
			Status: &commonpb.Status{ErrorCode: errorCode},
		}, err)
}

func (s *MockSuiteBase) setupHasPartition(collName string, partNames ...string) {
	s.mock.EXPECT().HasPartition(mock.Anything, mock.AnythingOfType("*milvuspb.HasPartitionRequest")).
		Call.Return(func(ctx context.Context, req *milvuspb.HasPartitionRequest) *milvuspb.BoolResponse {
		resp := &milvuspb.BoolResponse{Status: &commonpb.Status{}}
		if req.GetCollectionName() == collName {
			for _, partName := range partNames {
				if req.GetPartitionName() == partName {
					resp.Value = true
					break
				}
			}
		}
		return resp
	}, nil)
}

func (s *MockSuiteBase) setupHasPartitionError(errorCode commonpb.ErrorCode, err error) {
	s.mock.EXPECT().HasPartition(mock.Anything, mock.AnythingOfType("*milvuspb.HasPartitionRequest")).
		Return(&milvuspb.BoolResponse{
			Status: &commonpb.Status{ErrorCode: errorCode},
		}, err)
}

func (s *MockSuiteBase) setupDescribeCollection(_ string, schema *entity.Schema) {
	s.mock.EXPECT().DescribeCollection(mock.Anything, mock.AnythingOfType("*milvuspb.DescribeCollectionRequest")).
		Call.Return(func(ctx context.Context, req *milvuspb.DescribeCollectionRequest) *milvuspb.DescribeCollectionResponse {
		return &milvuspb.DescribeCollectionResponse{
			Status: &commonpb.Status{ErrorCode: commonpb.ErrorCode_Success},
			Schema: schema.ProtoMessage(),
		}
	}, nil)
}

func (s *MockSuiteBase) setupDescribeCollectionError(errorCode commonpb.ErrorCode, err error) {
	s.mock.EXPECT().DescribeCollection(mock.Anything, mock.AnythingOfType("*milvuspb.DescribeCollectionRequest")).
		Return(&milvuspb.DescribeCollectionResponse{
			Status: &commonpb.Status{ErrorCode: errorCode},
		}, err)
}

func (s *MockSuiteBase) getInt64FieldData(name string, data []int64) *schemapb.FieldData {
	return &schemapb.FieldData{
		Type:      schemapb.DataType_Int64,
		FieldName: name,
		Field: &schemapb.FieldData_Scalars{
			Scalars: &schemapb.ScalarField{
				Data: &schemapb.ScalarField_LongData{
					LongData: &schemapb.LongArray{
						Data: data,
					},
				},
			},
		},
	}
}

func (s *MockSuiteBase) getVarcharFieldData(name string, data []string) *schemapb.FieldData {
	return &schemapb.FieldData{
		Type:      schemapb.DataType_VarChar,
		FieldName: name,
		Field: &schemapb.FieldData_Scalars{
			Scalars: &schemapb.ScalarField{
				Data: &schemapb.ScalarField_StringData{
					StringData: &schemapb.StringArray{
						Data: data,
					},
				},
			},
		},
	}
}

func (s *MockSuiteBase) getJSONBytesFieldData(name string, data [][]byte, isDynamic bool) *schemapb.FieldData {
	return &schemapb.FieldData{
		Type:      schemapb.DataType_JSON,
		FieldName: name,
		Field: &schemapb.FieldData_Scalars{
			Scalars: &schemapb.ScalarField{
				Data: &schemapb.ScalarField_JsonData{
					JsonData: &schemapb.JSONArray{
						Data: data,
					},
				},
			},
		},
		IsDynamic: isDynamic,
	}
}

func (s *MockSuiteBase) getFloatVectorFieldData(name string, dim int64, data []float32) *schemapb.FieldData {
	return &schemapb.FieldData{
		Type:      schemapb.DataType_FloatVector,
		FieldName: name,
		Field: &schemapb.FieldData_Vectors{
			Vectors: &schemapb.VectorField{
				Dim: dim,
				Data: &schemapb.VectorField_FloatVector{
					FloatVector: &schemapb.FloatArray{
						Data: data,
					},
				},
			},
		},
	}
}

func (s *MockSuiteBase) getSuccessStatus() *commonpb.Status {
	return s.getStatus(commonpb.ErrorCode_Success, "")
}

func (s *MockSuiteBase) getStatus(code commonpb.ErrorCode, reason string) *commonpb.Status {
	return &commonpb.Status{
		ErrorCode: code,
		Reason:    reason,
	}
}

// ref https://stackoverflow.com/questions/42102496/testing-a-grpc-service

var errNotImplemented = errors.New("not implemented")

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
	MAlterCollectionField    ServiceMethod = 112
	MOperatePrivilegeV2      ServiceMethod = 113

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
	MAlterIndex            ServiceMethod = 406

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
	MSearchV2      ServiceMethod = 608

	MManualCompaction            ServiceMethod = 700
	MGetCompactionState          ServiceMethod = 701
	MGetCompactionStateWithPlans ServiceMethod = 702

	MGetPersistentSegmentInfo ServiceMethod = 800
	MGetQuerySegmentInfo      ServiceMethod = 801

	MGetComponentStates ServiceMethod = 900
	MGetVersion         ServiceMethod = 901
	MCheckHealth        ServiceMethod = 902

	MListDatabase     ServiceMethod = 1000
	MCreateDatabase   ServiceMethod = 1001
	MDropDatabase     ServiceMethod = 1002
	MAlterDatabase    ServiceMethod = 1003
	MDescribeDatabase ServiceMethod = 1004

	MReplicateMessage ServiceMethod = 1100
	MBackupRBAC       ServiceMethod = 1101
	MRestoreRBAC      ServiceMethod = 1102

	MCreatePrivilegeGroup  ServiceMethod = 1200
	MDropPrivilegeGroup    ServiceMethod = 1201
	MListPrivilegeGroups   ServiceMethod = 1202
	MOperatePrivilegeGroup ServiceMethod = 1203
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

// -- database --
// ListDatabases list all database in milvus cluster.
func (m *MockServer) ListDatabases(ctx context.Context, req *milvuspb.ListDatabasesRequest) (*milvuspb.ListDatabasesResponse, error) {
	f := m.GetInjection(MListDatabase)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*milvuspb.ListDatabasesResponse), err
	}
	r := &milvuspb.ListDatabasesResponse{}
	s, err := SuccessStatus()
	r.Status = s
	return r, err
}

// CreateDatabase create database with the given name.
func (m *MockServer) CreateDatabase(ctx context.Context, req *milvuspb.CreateDatabaseRequest) (*commonpb.Status, error) {
	f := m.GetInjection(MCreateDatabase)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*commonpb.Status), err
	}
	return SuccessStatus()
}

// DropDatabase drop database with the given db name.
func (m *MockServer) DropDatabase(ctx context.Context, req *milvuspb.DropDatabaseRequest) (*commonpb.Status, error) {
	f := m.GetInjection(MDropDatabase)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*commonpb.Status), err
	}
	return SuccessStatus()
}

func (m *MockServer) AlterDatabase(ctx context.Context, req *milvuspb.AlterDatabaseRequest) (*commonpb.Status, error) {
	f := m.GetInjection(MAlterDatabase)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*commonpb.Status), err
	}
	return SuccessStatus()
}

func (m *MockServer) DescribeDatabase(ctx context.Context, req *milvuspb.DescribeDatabaseRequest) (*milvuspb.DescribeDatabaseResponse, error) {
	f := m.GetInjection(MDescribeDatabase)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*milvuspb.DescribeDatabaseResponse), err
	}

	resp := &milvuspb.DescribeDatabaseResponse{
		Status: merr.Success(),
	}
	return resp, nil
}

func (m *MockServer) CreateCollection(ctx context.Context, req *milvuspb.CreateCollectionRequest) (*commonpb.Status, error) {
	f := m.GetInjection(MCreateCollection)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*commonpb.Status), err
	}
	return SuccessStatus()
}

func (m *MockServer) DropCollection(ctx context.Context, req *milvuspb.DropCollectionRequest) (*commonpb.Status, error) {
	f := m.GetInjection(MDropCollection)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*commonpb.Status), err
	}

	return SuccessStatus()
}

func (m *MockServer) HasCollection(ctx context.Context, req *milvuspb.HasCollectionRequest) (*milvuspb.BoolResponse, error) {
	f := m.GetInjection(MHasCollection)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*milvuspb.BoolResponse), err
	}
	s, err := SuccessStatus()
	return &milvuspb.BoolResponse{Status: s, Value: false}, err
}

func (m *MockServer) LoadCollection(ctx context.Context, req *milvuspb.LoadCollectionRequest) (*commonpb.Status, error) {
	f := m.GetInjection(MLoadCollection)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*commonpb.Status), err
	}
	return SuccessStatus()
}

func (m *MockServer) ReleaseCollection(ctx context.Context, req *milvuspb.ReleaseCollectionRequest) (*commonpb.Status, error) {
	f := m.GetInjection(MReleaseCollection)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*commonpb.Status), err
	}
	return SuccessStatus()
}

func (m *MockServer) DescribeCollection(ctx context.Context, req *milvuspb.DescribeCollectionRequest) (*milvuspb.DescribeCollectionResponse, error) {
	f := m.GetInjection(MDescribeCollection)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*milvuspb.DescribeCollectionResponse), err
	}
	r := &milvuspb.DescribeCollectionResponse{}
	s, err := SuccessStatus()
	r.Status = s
	return r, err
}

func (m *MockServer) GetCollectionStatistics(ctx context.Context, req *milvuspb.GetCollectionStatisticsRequest) (*milvuspb.GetCollectionStatisticsResponse, error) {
	f := m.GetInjection(MGetCollectionStatistics)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*milvuspb.GetCollectionStatisticsResponse), err
	}
	r := &milvuspb.GetCollectionStatisticsResponse{}
	s, err := SuccessStatus()
	r.Status = s
	return r, err
}

func (m *MockServer) ShowCollections(ctx context.Context, req *milvuspb.ShowCollectionsRequest) (*milvuspb.ShowCollectionsResponse, error) {
	f := m.GetInjection(MShowCollections)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*milvuspb.ShowCollectionsResponse), err
	}
	s, err := SuccessStatus()
	return &milvuspb.ShowCollectionsResponse{Status: s}, err
}

func (m *MockServer) AlterCollection(ctx context.Context, req *milvuspb.AlterCollectionRequest) (*commonpb.Status, error) {
	f := m.GetInjection(MAlterCollection)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*commonpb.Status), err
	}
	return SuccessStatus()
}

func (m *MockServer) OperatePrivilegeV2(ctx context.Context, req *milvuspb.OperatePrivilegeV2Request) (*commonpb.Status, error) {
	f := m.GetInjection(MOperatePrivilegeV2)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*commonpb.Status), err
	}
	return SuccessStatus()
}

func (m *MockServer) CreatePartition(ctx context.Context, req *milvuspb.CreatePartitionRequest) (*commonpb.Status, error) {
	f := m.GetInjection(MCreatePartition)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*commonpb.Status), err
	}
	return SuccessStatus()
}

func (m *MockServer) DropPartition(ctx context.Context, req *milvuspb.DropPartitionRequest) (*commonpb.Status, error) {
	f := m.GetInjection(MDropPartition)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*commonpb.Status), err
	}
	return SuccessStatus()
}

func (m *MockServer) HasPartition(ctx context.Context, req *milvuspb.HasPartitionRequest) (*milvuspb.BoolResponse, error) {
	f := m.GetInjection(MHasPartition)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*milvuspb.BoolResponse), err
	}
	s, err := SuccessStatus()
	return &milvuspb.BoolResponse{Status: s, Value: false}, err
}

func (m *MockServer) LoadPartitions(ctx context.Context, req *milvuspb.LoadPartitionsRequest) (*commonpb.Status, error) {
	f := m.GetInjection(MLoadPartitions)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*commonpb.Status), err
	}
	return SuccessStatus()
}

func (m *MockServer) ReleasePartitions(ctx context.Context, req *milvuspb.ReleasePartitionsRequest) (*commonpb.Status, error) {
	f := m.GetInjection(MReleasePartitions)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*commonpb.Status), err
	}
	return SuccessStatus()
}

func (m *MockServer) GetPartitionStatistics(_ context.Context, _ *milvuspb.GetPartitionStatisticsRequest) (*milvuspb.GetPartitionStatisticsResponse, error) {
	panic("not implemented") // TODO: Implement
}

func (m *MockServer) ShowPartitions(ctx context.Context, req *milvuspb.ShowPartitionsRequest) (*milvuspb.ShowPartitionsResponse, error) {
	f := m.GetInjection(MShowPartitions)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*milvuspb.ShowPartitionsResponse), err
	}
	s, err := SuccessStatus()
	return &milvuspb.ShowPartitionsResponse{Status: s}, err
}

func (m *MockServer) GetLoadingProgress(ctx context.Context, req *milvuspb.GetLoadingProgressRequest) (*milvuspb.GetLoadingProgressResponse, error) {
	f := m.GetInjection(MGetLoadingProgress)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*milvuspb.GetLoadingProgressResponse), err
	}
	s, err := SuccessStatus()
	return &milvuspb.GetLoadingProgressResponse{Status: s}, err
}

func (m *MockServer) GetLoadState(ctx context.Context, req *milvuspb.GetLoadStateRequest) (*milvuspb.GetLoadStateResponse, error) {
	f := m.GetInjection(MGetLoadState)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*milvuspb.GetLoadStateResponse), err
	}
	s, err := SuccessStatus()
	return &milvuspb.GetLoadStateResponse{Status: s}, err
}

func (m *MockServer) CreateIndex(ctx context.Context, req *milvuspb.CreateIndexRequest) (*commonpb.Status, error) {
	f := m.GetInjection(MCreateIndex)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*commonpb.Status), err
	}
	return SuccessStatus()
}

func (m *MockServer) AlterIndex(ctx context.Context, req *milvuspb.AlterIndexRequest) (*commonpb.Status, error) {
	f := m.GetInjection(MAlterIndex)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*commonpb.Status), err
	}
	return SuccessStatus()
}

func (m *MockServer) DescribeIndex(ctx context.Context, req *milvuspb.DescribeIndexRequest) (*milvuspb.DescribeIndexResponse, error) {
	f := m.GetInjection(MDescribeIndex)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*milvuspb.DescribeIndexResponse), err
	}
	s, err := SuccessStatus()
	return &milvuspb.DescribeIndexResponse{Status: s}, err
}

func (m *MockServer) GetIndexState(ctx context.Context, req *milvuspb.GetIndexStateRequest) (*milvuspb.GetIndexStateResponse, error) {
	f := m.GetInjection(MGetIndexState)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*milvuspb.GetIndexStateResponse), err
	}
	s, err := SuccessStatus()
	return &milvuspb.GetIndexStateResponse{Status: s}, err
}

func (m *MockServer) GetIndexBuildProgress(ctx context.Context, req *milvuspb.GetIndexBuildProgressRequest) (*milvuspb.GetIndexBuildProgressResponse, error) {
	f := m.GetInjection(MGetIndexBuildProgress)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*milvuspb.GetIndexBuildProgressResponse), err
	}
	s, err := SuccessStatus()
	return &milvuspb.GetIndexBuildProgressResponse{Status: s}, err
}

func (m *MockServer) DropIndex(ctx context.Context, req *milvuspb.DropIndexRequest) (*commonpb.Status, error) {
	f := m.GetInjection(MDropIndex)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*commonpb.Status), err
	}
	return SuccessStatus()
}

func (m *MockServer) Insert(ctx context.Context, req *milvuspb.InsertRequest) (*milvuspb.MutationResult, error) {
	f := m.GetInjection(MInsert)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*milvuspb.MutationResult), err
	}
	s, err := SuccessStatus()
	return &milvuspb.MutationResult{Status: s}, err
}

func (m *MockServer) Search(ctx context.Context, req *milvuspb.SearchRequest) (*milvuspb.SearchResults, error) {
	f := m.GetInjection(MSearch)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*milvuspb.SearchResults), err
	}
	s, err := SuccessStatus()
	return &milvuspb.SearchResults{Status: s}, err
}

func (m *MockServer) Flush(ctx context.Context, req *milvuspb.FlushRequest) (*milvuspb.FlushResponse, error) {
	f := m.GetInjection(MFlush)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*milvuspb.FlushResponse), err
	}
	s, err := SuccessStatus()
	return &milvuspb.FlushResponse{Status: s}, err
}

func (m *MockServer) Query(ctx context.Context, req *milvuspb.QueryRequest) (*milvuspb.QueryResults, error) {
	f := m.GetInjection(MQuery)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*milvuspb.QueryResults), err
	}
	s, err := SuccessStatus()
	return &milvuspb.QueryResults{Status: s}, err
}

func (m *MockServer) GetPersistentSegmentInfo(ctx context.Context, req *milvuspb.GetPersistentSegmentInfoRequest) (*milvuspb.GetPersistentSegmentInfoResponse, error) {
	f := m.GetInjection(MGetPersistentSegmentInfo)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*milvuspb.GetPersistentSegmentInfoResponse), err
	}
	s, err := SuccessStatus()
	return &milvuspb.GetPersistentSegmentInfoResponse{Status: s}, err
}

func (m *MockServer) GetQuerySegmentInfo(ctx context.Context, req *milvuspb.GetQuerySegmentInfoRequest) (*milvuspb.GetQuerySegmentInfoResponse, error) {
	f := m.GetInjection(MGetQuerySegmentInfo)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*milvuspb.GetQuerySegmentInfoResponse), err
	}
	s, err := SuccessStatus()
	return &milvuspb.GetQuerySegmentInfoResponse{Status: s}, err
}

func (m *MockServer) Dummy(_ context.Context, _ *milvuspb.DummyRequest) (*milvuspb.DummyResponse, error) {
	panic("not implemented") // TODO: Implement
}

// TODO: remove
func (m *MockServer) RegisterLink(_ context.Context, _ *milvuspb.RegisterLinkRequest) (*milvuspb.RegisterLinkResponse, error) {
	panic("not implemented") // TODO: Implement
}

func (m *MockServer) CalcDistance(ctx context.Context, req *milvuspb.CalcDistanceRequest) (*milvuspb.CalcDistanceResults, error) {
	f := m.GetInjection(MCalcDistance)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*milvuspb.CalcDistanceResults), err
	}
	resp := &milvuspb.CalcDistanceResults{}
	s, err := SuccessStatus()
	resp.Status = s
	return resp, err
}

func (m *MockServer) Delete(ctx context.Context, req *milvuspb.DeleteRequest) (*milvuspb.MutationResult, error) {
	f := m.GetInjection(MDelete)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*milvuspb.MutationResult), err
	}
	resp := &milvuspb.MutationResult{}
	s, err := SuccessStatus()
	resp.Status = s
	return resp, err
}

// https://wiki.lfaidata.foundation/display/MIL/MEP+8+--+Add+metrics+for+proxy
func (m *MockServer) GetMetrics(_ context.Context, _ *milvuspb.GetMetricsRequest) (*milvuspb.GetMetricsResponse, error) {
	panic("not implemented") // TODO: Implement
}

func (m *MockServer) CreateAlias(ctx context.Context, req *milvuspb.CreateAliasRequest) (*commonpb.Status, error) {
	f := m.GetInjection(MCreateAlias)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*commonpb.Status), err
	}
	return SuccessStatus()
}

func (m *MockServer) DropAlias(ctx context.Context, req *milvuspb.DropAliasRequest) (*commonpb.Status, error) {
	f := m.GetInjection(MDropAlias)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*commonpb.Status), err
	}
	return SuccessStatus()
}

func (m *MockServer) AlterAlias(ctx context.Context, req *milvuspb.AlterAliasRequest) (*commonpb.Status, error) {
	f := m.GetInjection(MAlterAlias)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*commonpb.Status), err
	}

	return SuccessStatus()
}

func (m *MockServer) GetFlushState(ctx context.Context, req *milvuspb.GetFlushStateRequest) (*milvuspb.GetFlushStateResponse, error) {
	f := m.GetInjection(MGetFlushState)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*milvuspb.GetFlushStateResponse), err
	}

	resp := &milvuspb.GetFlushStateResponse{}
	s, err := SuccessStatus()
	resp.Status = s
	return resp, err
}

func (m *MockServer) LoadBalance(_ context.Context, _ *milvuspb.LoadBalanceRequest) (*commonpb.Status, error) {
	panic("not implemented") // TODO: Implement
}

func (m *MockServer) GetCompactionState(ctx context.Context, req *milvuspb.GetCompactionStateRequest) (*milvuspb.GetCompactionStateResponse, error) {
	f := m.GetInjection(MGetCompactionState)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*milvuspb.GetCompactionStateResponse), err
	}

	resp := &milvuspb.GetCompactionStateResponse{}
	s, err := SuccessStatus()
	resp.Status = s
	return resp, err
}

func (m *MockServer) ManualCompaction(ctx context.Context, req *milvuspb.ManualCompactionRequest) (*milvuspb.ManualCompactionResponse, error) {
	f := m.GetInjection(MManualCompaction)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*milvuspb.ManualCompactionResponse), err
	}
	resp := &milvuspb.ManualCompactionResponse{}
	s, err := SuccessStatus()
	resp.Status = s
	return resp, err
}

func (m *MockServer) GetCompactionStateWithPlans(ctx context.Context, req *milvuspb.GetCompactionPlansRequest) (*milvuspb.GetCompactionPlansResponse, error) {
	f := m.GetInjection(MGetCompactionStateWithPlans)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*milvuspb.GetCompactionPlansResponse), err
	}

	resp := &milvuspb.GetCompactionPlansResponse{}
	s, err := SuccessStatus()
	resp.Status = s
	return resp, err
}

func (m *MockServer) GetReplicas(ctx context.Context, req *milvuspb.GetReplicasRequest) (*milvuspb.GetReplicasResponse, error) {
	f := m.GetInjection(MGetReplicas)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*milvuspb.GetReplicasResponse), err
	}
	resp := &milvuspb.GetReplicasResponse{}
	s, err := SuccessStatus()
	resp.Status = s
	return resp, err
}

// https://wiki.lfaidata.foundation/display/MIL/MEP+24+--+Support+bulk+load
func (m *MockServer) Import(_ context.Context, _ *milvuspb.ImportRequest) (*milvuspb.ImportResponse, error) {
	panic("not implemented") // TODO: Implement
}

func (m *MockServer) GetImportState(_ context.Context, _ *milvuspb.GetImportStateRequest) (*milvuspb.GetImportStateResponse, error) {
	panic("not implemented") // TODO: Implement
}

func (m *MockServer) ListImportTasks(_ context.Context, _ *milvuspb.ListImportTasksRequest) (*milvuspb.ListImportTasksResponse, error) {
	panic("not implemented") // TODO: Implement
}

// https://wiki.lfaidata.foundation/display/MIL/MEP+27+--+Support+Basic+Authentication
func (m *MockServer) CreateCredential(ctx context.Context, req *milvuspb.CreateCredentialRequest) (*commonpb.Status, error) {
	f := m.GetInjection(MCreateCredential)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*commonpb.Status), err
	}
	return SuccessStatus()
}

func (m *MockServer) UpdateCredential(ctx context.Context, req *milvuspb.UpdateCredentialRequest) (*commonpb.Status, error) {
	f := m.GetInjection(MUpdateCredential)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*commonpb.Status), err
	}
	return SuccessStatus()
}

func (m *MockServer) DeleteCredential(ctx context.Context, req *milvuspb.DeleteCredentialRequest) (*commonpb.Status, error) {
	f := m.GetInjection(MDeleteCredential)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*commonpb.Status), err
	}
	return SuccessStatus()
}

func (m *MockServer) ListCredUsers(ctx context.Context, req *milvuspb.ListCredUsersRequest) (*milvuspb.ListCredUsersResponse, error) {
	f := m.GetInjection(MListCredUsers)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*milvuspb.ListCredUsersResponse), err
	}
	resp := &milvuspb.ListCredUsersResponse{}
	s, err := SuccessStatus()
	resp.Status = s
	return resp, err
}

// https://wiki.lfaidata.foundation/display/MIL/MEP+29+--+Support+Role-Based+Access+Control
func (m *MockServer) CreateRole(_ context.Context, _ *milvuspb.CreateRoleRequest) (*commonpb.Status, error) {
	panic("not implemented") // TODO: Implement
}

func (m *MockServer) DropRole(_ context.Context, _ *milvuspb.DropRoleRequest) (*commonpb.Status, error) {
	panic("not implemented") // TODO: Implement
}

func (m *MockServer) OperateUserRole(_ context.Context, _ *milvuspb.OperateUserRoleRequest) (*commonpb.Status, error) {
	panic("not implemented") // TODO: Implement
}

func (m *MockServer) SelectRole(_ context.Context, _ *milvuspb.SelectRoleRequest) (*milvuspb.SelectRoleResponse, error) {
	panic("not implemented") // TODO: Implement
}

func (m *MockServer) SelectUser(_ context.Context, _ *milvuspb.SelectUserRequest) (*milvuspb.SelectUserResponse, error) {
	panic("not implemented") // TODO: Implement
}

func (m *MockServer) OperatePrivilege(_ context.Context, _ *milvuspb.OperatePrivilegeRequest) (*commonpb.Status, error) {
	panic("not implemented") // TODO: Implement
}

func (m *MockServer) SelectGrant(_ context.Context, _ *milvuspb.SelectGrantRequest) (*milvuspb.SelectGrantResponse, error) {
	panic("not implemented") // TODO: Implement
}

func (m *MockServer) CreateResourceGroup(_ context.Context, _ *milvuspb.CreateResourceGroupRequest) (*commonpb.Status, error) {
	panic("not implemented") // TODO: Implement
}

func (m *MockServer) DropResourceGroup(_ context.Context, _ *milvuspb.DropResourceGroupRequest) (*commonpb.Status, error) {
	panic("not implemented") // TODO: Implement
}

func (m *MockServer) TransferNode(_ context.Context, _ *milvuspb.TransferNodeRequest) (*commonpb.Status, error) {
	panic("not implemented") // TODO: Implement
}

func (m *MockServer) TransferReplica(_ context.Context, _ *milvuspb.TransferReplicaRequest) (*commonpb.Status, error) {
	panic("not implemented") // TODO: Implement
}

func (m *MockServer) ListResourceGroups(_ context.Context, _ *milvuspb.ListResourceGroupsRequest) (*milvuspb.ListResourceGroupsResponse, error) {
	panic("not implemented") // TODO: Implement
}

func (m *MockServer) DescribeResourceGroup(_ context.Context, _ *milvuspb.DescribeResourceGroupRequest) (*milvuspb.DescribeResourceGroupResponse, error) {
	panic("not implemented") // TODO: Implement
}

func (m *MockServer) RenameCollection(_ context.Context, _ *milvuspb.RenameCollectionRequest) (*commonpb.Status, error) {
	panic("not implemented") // TODO: Implement
}

func (m *MockServer) DescribeAlias(_ context.Context, _ *milvuspb.DescribeAliasRequest) (*milvuspb.DescribeAliasResponse, error) {
	panic("not implemented") // TODO: Implement
}

func (m *MockServer) ListAliases(_ context.Context, _ *milvuspb.ListAliasesRequest) (*milvuspb.ListAliasesResponse, error) {
	panic("not implemented") // TODO: Implement
}

func (m *MockServer) FlushAll(_ context.Context, _ *milvuspb.FlushAllRequest) (*milvuspb.FlushAllResponse, error) {
	panic("not implemented") // TODO: Implement
}

func (m *MockServer) GetFlushAllState(_ context.Context, _ *milvuspb.GetFlushAllStateRequest) (*milvuspb.GetFlushAllStateResponse, error) {
	panic("not implemented") // TODO: Implement
}

func (m *MockServer) ListIndexedSegment(_ context.Context, _ *federpb.ListIndexedSegmentRequest) (*federpb.ListIndexedSegmentResponse, error) {
	panic("not implemented") // TODO: Implement
}

func (m *MockServer) DescribeSegmentIndexData(_ context.Context, _ *federpb.DescribeSegmentIndexDataRequest) (*federpb.DescribeSegmentIndexDataResponse, error) {
	panic("not implemented") // TODO: Implement
}

func (m *MockServer) GetIndexStatistics(_ context.Context, _ *milvuspb.GetIndexStatisticsRequest) (*milvuspb.GetIndexStatisticsResponse, error) {
	panic("not implemented") // TODO: Implement
}

func (m *MockServer) AllocTimestamp(_ context.Context, _ *milvuspb.AllocTimestampRequest) (*milvuspb.AllocTimestampResponse, error) {
	panic("not implemented")
}

func (m *MockServer) ReplicateMessage(ctx context.Context, req *milvuspb.ReplicateMessageRequest) (*milvuspb.ReplicateMessageResponse, error) {
	f := m.GetInjection(MReplicateMessage)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*milvuspb.ReplicateMessageResponse), err
	}
	s, err := SuccessStatus()
	return &milvuspb.ReplicateMessageResponse{Status: s}, err
}

func (m *MockServer) Connect(_ context.Context, _ *milvuspb.ConnectRequest) (*milvuspb.ConnectResponse, error) {
	return &milvuspb.ConnectResponse{
		Status:     &commonpb.Status{ErrorCode: commonpb.ErrorCode_Success},
		Identifier: 1,
	}, nil
}

//func (m *MockServer) DescribePartition(ctx context.Context, req *milvuspb.DescribePartitionRequest) (*milvuspb.DescribePartitionResponse, error) {
//	panic("not implemented") // TODO: Implement
//}

func (m *MockServer) GetComponentStates(ctx context.Context, req *milvuspb.GetComponentStatesRequest) (*milvuspb.ComponentStates, error) {
	f := m.GetInjection(MGetComponentStates)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*milvuspb.ComponentStates), err
	}
	s, err := SuccessStatus()
	return &milvuspb.ComponentStates{Status: s}, err
}

func (m *MockServer) GetVersion(ctx context.Context, req *milvuspb.GetVersionRequest) (*milvuspb.GetVersionResponse, error) {
	f := m.GetInjection(MGetVersion)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*milvuspb.GetVersionResponse), err
	}
	s, err := SuccessStatus()
	return &milvuspb.GetVersionResponse{Status: s}, err
}

func (m *MockServer) CheckHealth(ctx context.Context, req *milvuspb.CheckHealthRequest) (*milvuspb.CheckHealthResponse, error) {
	f := m.GetInjection(MCheckHealth)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*milvuspb.CheckHealthResponse), err
	}
	s, err := SuccessStatus()
	return &milvuspb.CheckHealthResponse{Status: s}, err
}

func getSuccessStatus() *commonpb.Status {
	return &commonpb.Status{ErrorCode: commonpb.ErrorCode_Success}
}

func SuccessStatus() (*commonpb.Status, error) {
	return &commonpb.Status{ErrorCode: commonpb.ErrorCode_Success}, nil
}

func BadRequestStatus() (*commonpb.Status, error) {
	return &commonpb.Status{ErrorCode: commonpb.ErrorCode_IllegalArgument}, errors.New("illegal request type")
}

func BadStatus() (*commonpb.Status, error) {
	return &commonpb.Status{
		ErrorCode: commonpb.ErrorCode_UnexpectedError,
		Reason:    "fail reason",
	}, nil
}

func (m *MockServer) Upsert(ctx context.Context, req *milvuspb.UpsertRequest) (*milvuspb.MutationResult, error) {
	f := m.GetInjection(MUpsert)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*milvuspb.MutationResult), err
	}
	s, err := SuccessStatus()
	return &milvuspb.MutationResult{Status: s}, err
}

func (m *MockServer) HybridSearch(ctx context.Context, req *milvuspb.HybridSearchRequest) (*milvuspb.SearchResults, error) {
	f := m.GetInjection(MSearchV2)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*milvuspb.SearchResults), err
	}
	status, err := SuccessStatus()
	return &milvuspb.SearchResults{Status: status}, err
}

func (m *MockServer) UpdateResourceGroups(_ context.Context, _ *milvuspb.UpdateResourceGroupsRequest) (*commonpb.Status, error) {
	return nil, errors.New("not implemented")
}

func (m *MockServer) BackupRBAC(ctx context.Context, req *milvuspb.BackupRBACMetaRequest) (*milvuspb.BackupRBACMetaResponse, error) {
	f := m.GetInjection(MBackupRBAC)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*milvuspb.BackupRBACMetaResponse), err
	}
	s, err := SuccessStatus()
	return &milvuspb.BackupRBACMetaResponse{Status: s}, err
}

func (m *MockServer) RestoreRBAC(ctx context.Context, req *milvuspb.RestoreRBACMetaRequest) (*commonpb.Status, error) {
	f := m.GetInjection(MRestoreRBAC)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*commonpb.Status), err
	}
	return SuccessStatus()
}

func (m *MockServer) AlterCollectionField(ctx context.Context, req *milvuspb.AlterCollectionFieldRequest) (*commonpb.Status, error) {
	f := m.GetInjection(MAlterCollectionField)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*commonpb.Status), err
	}
	return SuccessStatus()
}

func (m *MockServer) CreatePrivilegeGroup(ctx context.Context, req *milvuspb.CreatePrivilegeGroupRequest) (*commonpb.Status, error) {
	f := m.GetInjection(MCreatePrivilegeGroup)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*commonpb.Status), err
	}
	return SuccessStatus()
}

func (m *MockServer) DropPrivilegeGroup(ctx context.Context, req *milvuspb.DropPrivilegeGroupRequest) (*commonpb.Status, error) {
	f := m.GetInjection(MDropPrivilegeGroup)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*commonpb.Status), err
	}
	return SuccessStatus()
}

func (m *MockServer) ListPrivilegeGroups(ctx context.Context, req *milvuspb.ListPrivilegeGroupsRequest) (*milvuspb.ListPrivilegeGroupsResponse, error) {
	f := m.GetInjection(MListPrivilegeGroups)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*milvuspb.ListPrivilegeGroupsResponse), err
	}
	s, err := SuccessStatus()
	return &milvuspb.ListPrivilegeGroupsResponse{Status: s}, err
}

func (m *MockServer) OperatePrivilegeGroup(ctx context.Context, req *milvuspb.OperatePrivilegeGroupRequest) (*commonpb.Status, error) {
	f := m.GetInjection(MOperatePrivilegeGroup)
	if f != nil {
		r, err := f(ctx, req)
		return r.(*commonpb.Status), err
	}
	return SuccessStatus()
}
