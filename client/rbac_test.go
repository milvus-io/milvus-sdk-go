package client

import (
	"context"
	"errors"
	"net"
	"testing"

	"github.com/milvus-io/milvus-sdk-go/v2/entity"
	"github.com/milvus-io/milvus-sdk-go/v2/mocks"
	"github.com/stretchr/testify/suite"
	"google.golang.org/grpc"
	"google.golang.org/grpc/test/bufconn"

	tmock "github.com/stretchr/testify/mock"

	common "github.com/milvus-io/milvus-proto/go-api/commonpb"
	server "github.com/milvus-io/milvus-proto/go-api/milvuspb"
)

type RBACSuite struct {
	suite.Suite

	lis  *bufconn.Listener
	svr  *grpc.Server
	mock *mocks.MilvusServiceServer

	client Client
}

func (s *RBACSuite) SetupSuite() {
	s.lis = bufconn.Listen(bufSize)
	s.svr = grpc.NewServer()

	s.mock = &mocks.MilvusServiceServer{}

	server.RegisterMilvusServiceServer(s.svr, s.mock)

	go func() {
		s.T().Log("start mock server")
		if err := s.svr.Serve(s.lis); err != nil {
			s.Fail("failed to server mock server", err.Error())
		}
	}()
}

func (s *RBACSuite) TearDownSuite() {
	s.svr.Stop()
	s.lis.Close()
}

func (s *RBACSuite) mockDialer(context.Context, string) (net.Conn, error) {
	return s.lis.Dial()
}

func (s *RBACSuite) SetupTest() {
	c, err := NewClient(context.Background(), Config{
		Address: "bufnet2",
		DialOptions: []grpc.DialOption{
			grpc.WithBlock(),
			grpc.WithInsecure(),
			grpc.WithContextDialer(s.mockDialer),
		},
	})
	s.Require().NoError(err)

	s.client = c
}

func (s *RBACSuite) TearDownTest() {
	s.client.Close()
	s.client = nil
}

func (s *RBACSuite) TestCreateRole() {
	ctx := context.Background()
	roleName := "testRole"

	s.Run("normal run", func() {
		ctx, cancel := context.WithCancel(ctx)
		defer cancel()
		s.mock.ExpectedCalls = nil
		s.mock.EXPECT().CreateRole(tmock.Anything, tmock.Anything).Run(func(ctx context.Context, req *server.CreateRoleRequest) {
			s.Equal(roleName, req.GetEntity().GetName())
		}).Return(&common.Status{
			ErrorCode: common.ErrorCode_Success,
		}, nil)
		err := s.client.CreateRole(ctx, roleName)

		s.NoError(err)
	})

	s.Run("rpc error", func() {
		ctx, cancel := context.WithCancel(ctx)
		defer cancel()
		s.mock.ExpectedCalls = nil
		s.mock.EXPECT().CreateRole(tmock.Anything, tmock.Anything).Return(nil, errors.New("mock error"))
		err := s.client.CreateRole(ctx, roleName)

		s.Error(err)
	})

	s.Run("status error", func() {
		ctx, cancel := context.WithCancel(ctx)
		defer cancel()
		s.mock.ExpectedCalls = nil
		s.mock.EXPECT().CreateRole(tmock.Anything, tmock.Anything).Return(&common.Status{
			ErrorCode: common.ErrorCode_UnexpectedError,
		}, nil)
		err := s.client.CreateRole(ctx, roleName)

		s.Error(err)
	})

	s.Run("service not ready", func() {
		ctx, cancel := context.WithCancel(ctx)
		defer cancel()

		c := &GrpcClient{}
		err := c.CreateRole(ctx, roleName)
		s.Error(err)
		s.ErrorIs(err, ErrClientNotReady)
	})
}

func (s *RBACSuite) TestDropRole() {
	ctx := context.Background()
	roleName := "testRole"

	s.Run("normal run", func() {
		ctx, cancel := context.WithCancel(ctx)
		defer cancel()
		s.mock.ExpectedCalls = nil
		s.mock.EXPECT().DropRole(tmock.Anything, tmock.Anything).Run(func(ctx context.Context, req *server.DropRoleRequest) {
			s.Equal(roleName, req.GetRoleName())
		}).Return(&common.Status{
			ErrorCode: common.ErrorCode_Success,
		}, nil)
		err := s.client.DropRole(ctx, roleName)

		s.NoError(err)
	})

	s.Run("rpc error", func() {
		ctx, cancel := context.WithCancel(ctx)
		defer cancel()
		s.mock.ExpectedCalls = nil
		s.mock.EXPECT().DropRole(tmock.Anything, tmock.Anything).Return(nil, errors.New("mock error"))
		err := s.client.DropRole(ctx, roleName)

		s.Error(err)
	})

	s.Run("status error", func() {
		ctx, cancel := context.WithCancel(ctx)
		defer cancel()
		s.mock.ExpectedCalls = nil
		s.mock.EXPECT().DropRole(tmock.Anything, tmock.Anything).Return(&common.Status{
			ErrorCode: common.ErrorCode_UnexpectedError,
		}, nil)
		err := s.client.DropRole(ctx, roleName)

		s.Error(err)
	})

	s.Run("service not ready", func() {
		ctx, cancel := context.WithCancel(ctx)
		defer cancel()

		c := &GrpcClient{}
		err := c.DropRole(ctx, roleName)
		s.Error(err)
		s.ErrorIs(err, ErrClientNotReady)
	})
}

func (s *RBACSuite) TestAddUserRole() {
	ctx := context.Background()
	username := "testUser"
	roleName := "testRole"

	s.Run("normal run", func() {
		ctx, cancel := context.WithCancel(ctx)
		defer cancel()
		s.mock.ExpectedCalls = nil
		s.mock.EXPECT().OperateUserRole(tmock.Anything, tmock.Anything).Run(func(ctx context.Context, req *server.OperateUserRoleRequest) {
			s.Equal(server.OperateUserRoleType_AddUserToRole, req.GetType())
			s.Equal(username, req.GetUsername())
			s.Equal(roleName, req.GetRoleName())
		}).Return(&common.Status{
			ErrorCode: common.ErrorCode_Success,
		}, nil)
		err := s.client.AddUserRole(ctx, username, roleName)

		s.NoError(err)
	})

	s.Run("rpc error", func() {
		ctx, cancel := context.WithCancel(ctx)
		defer cancel()
		s.mock.ExpectedCalls = nil
		s.mock.EXPECT().OperateUserRole(tmock.Anything, tmock.Anything).Return(nil, errors.New("mock error"))
		err := s.client.AddUserRole(ctx, username, roleName)

		s.Error(err)
	})

	s.Run("status error", func() {
		ctx, cancel := context.WithCancel(ctx)
		defer cancel()
		s.mock.ExpectedCalls = nil
		s.mock.EXPECT().OperateUserRole(tmock.Anything, tmock.Anything).Return(&common.Status{
			ErrorCode: common.ErrorCode_UnexpectedError,
		}, nil)
		err := s.client.AddUserRole(ctx, username, roleName)

		s.Error(err)
	})

	s.Run("service not ready", func() {
		ctx, cancel := context.WithCancel(ctx)
		defer cancel()

		c := &GrpcClient{}
		err := c.AddUserRole(ctx, username, roleName)
		s.Error(err)
		s.ErrorIs(err, ErrClientNotReady)
	})
}

func (s *RBACSuite) TestRemoveUserRole() {
	ctx := context.Background()
	username := "testUser"
	roleName := "testRole"

	s.Run("normal run", func() {
		ctx, cancel := context.WithCancel(ctx)
		defer cancel()
		s.mock.ExpectedCalls = nil
		s.mock.EXPECT().OperateUserRole(tmock.Anything, tmock.Anything).Run(func(ctx context.Context, req *server.OperateUserRoleRequest) {
			s.Equal(server.OperateUserRoleType_RemoveUserFromRole, req.GetType())
			s.Equal(username, req.GetUsername())
			s.Equal(roleName, req.GetRoleName())
		}).Return(&common.Status{
			ErrorCode: common.ErrorCode_Success,
		}, nil)
		err := s.client.RemoveUserRole(ctx, username, roleName)

		s.NoError(err)
	})

	s.Run("rpc error", func() {
		ctx, cancel := context.WithCancel(ctx)
		defer cancel()
		s.mock.ExpectedCalls = nil
		s.mock.EXPECT().OperateUserRole(tmock.Anything, tmock.Anything).Return(nil, errors.New("mock error"))
		err := s.client.RemoveUserRole(ctx, username, roleName)

		s.Error(err)
	})

	s.Run("status error", func() {
		ctx, cancel := context.WithCancel(ctx)
		defer cancel()
		s.mock.ExpectedCalls = nil
		s.mock.EXPECT().OperateUserRole(tmock.Anything, tmock.Anything).Return(&common.Status{
			ErrorCode: common.ErrorCode_UnexpectedError,
		}, nil)
		err := s.client.RemoveUserRole(ctx, username, roleName)

		s.Error(err)
	})

	s.Run("service not ready", func() {
		ctx, cancel := context.WithCancel(ctx)
		defer cancel()

		c := &GrpcClient{}
		err := c.RemoveUserRole(ctx, username, roleName)
		s.Error(err)
		s.ErrorIs(err, ErrClientNotReady)
	})
}

func (s *RBACSuite) TestListRoles() {
	ctx := context.Background()
	roleName := "testRole"

	s.Run("normal run", func() {
		ctx, cancel := context.WithCancel(ctx)
		defer cancel()
		s.mock.ExpectedCalls = nil
		s.mock.EXPECT().SelectRole(tmock.Anything, tmock.Anything).Run(func(ctx context.Context, req *server.SelectRoleRequest) {
			s.False(req.GetIncludeUserInfo())
		}).Return(&server.SelectRoleResponse{
			Status: &common.Status{ErrorCode: common.ErrorCode_Success},
			Results: []*server.RoleResult{
				{
					Role: &server.RoleEntity{
						Name: roleName,
					},
				},
			},
		}, nil)
		roles, err := s.client.ListRoles(ctx)

		s.NoError(err)
		s.ElementsMatch([]entity.Role{{Name: roleName}}, roles)
	})

	s.Run("rpc error", func() {
		ctx, cancel := context.WithCancel(ctx)
		defer cancel()
		s.mock.ExpectedCalls = nil
		s.mock.EXPECT().SelectRole(tmock.Anything, tmock.Anything).Return(nil, errors.New("mock error"))

		_, err := s.client.ListRoles(ctx)
		s.Error(err)
	})

	s.Run("status error", func() {
		ctx, cancel := context.WithCancel(ctx)
		defer cancel()
		s.mock.ExpectedCalls = nil
		s.mock.EXPECT().SelectRole(tmock.Anything, tmock.Anything).Return(&server.SelectRoleResponse{
			Status: &common.Status{ErrorCode: common.ErrorCode_UnexpectedError},
		}, nil)

		_, err := s.client.ListRoles(ctx)
		s.Error(err)
	})

	s.Run("service not ready", func() {
		ctx, cancel := context.WithCancel(ctx)
		defer cancel()

		c := &GrpcClient{}
		_, err := c.ListRoles(ctx)
		s.Error(err)
		s.ErrorIs(err, ErrClientNotReady)
	})
}

func (s *RBACSuite) TestListUser() {
	ctx := context.Background()
	userName := "testUser"

	s.Run("normal run", func() {
		ctx, cancel := context.WithCancel(ctx)
		defer cancel()
		s.mock.ExpectedCalls = nil
		s.mock.EXPECT().SelectUser(tmock.Anything, tmock.Anything).Run(func(ctx context.Context, req *server.SelectUserRequest) {
			s.False(req.GetIncludeRoleInfo())
		}).Return(&server.SelectUserResponse{
			Status: &common.Status{ErrorCode: common.ErrorCode_Success},
			Results: []*server.UserResult{
				{
					User: &server.UserEntity{
						Name: userName,
					},
				},
			},
		}, nil)
		users, err := s.client.ListUsers(ctx)

		s.NoError(err)
		s.ElementsMatch([]entity.User{{Name: userName}}, users)
	})

	s.Run("rpc error", func() {
		ctx, cancel := context.WithCancel(ctx)
		defer cancel()
		s.mock.ExpectedCalls = nil
		s.mock.EXPECT().SelectUser(tmock.Anything, tmock.Anything).Return(nil, errors.New("mock error"))

		_, err := s.client.ListUsers(ctx)
		s.Error(err)
	})

	s.Run("status error", func() {
		ctx, cancel := context.WithCancel(ctx)
		defer cancel()
		s.mock.ExpectedCalls = nil
		s.mock.EXPECT().SelectUser(tmock.Anything, tmock.Anything).Return(&server.SelectUserResponse{
			Status: &common.Status{ErrorCode: common.ErrorCode_UnexpectedError},
		}, nil)

		_, err := s.client.ListUsers(ctx)
		s.Error(err)
	})

	s.Run("service not ready", func() {
		ctx, cancel := context.WithCancel(ctx)
		defer cancel()

		c := &GrpcClient{}
		_, err := c.ListUsers(ctx)
		s.Error(err)
		s.ErrorIs(err, ErrClientNotReady)
	})
}

func (s *RBACSuite) TestGrant() {
	ctx := context.Background()

	roleName := "testRole"
	objectName := testCollectionName
	objectType := entity.PriviledegeObjectTypeCollection

	s.Run("normal run", func() {
		ctx, cancel := context.WithCancel(ctx)
		defer cancel()
		s.mock.ExpectedCalls = nil
		s.mock.EXPECT().OperatePrivilege(tmock.Anything, tmock.Anything).Run(func(ctx context.Context, req *server.OperatePrivilegeRequest) {
			s.Equal(roleName, req.GetEntity().GetRole().GetName())
			s.Equal(objectName, req.GetEntity().GetObjectName())
			s.Equal(common.ObjectType_name[int32(objectType)], req.GetEntity().GetObject().GetName())
			s.Equal(server.OperatePrivilegeType_Grant, req.GetType())
		}).Return(&common.Status{ErrorCode: common.ErrorCode_Success}, nil)

		err := s.client.Grant(ctx, roleName, objectType, objectName)

		s.NoError(err)
	})

	s.Run("rpc error", func() {
		ctx, cancel := context.WithCancel(ctx)
		defer cancel()
		s.mock.ExpectedCalls = nil
		s.mock.EXPECT().OperatePrivilege(tmock.Anything, tmock.Anything).Return(nil, errors.New("mock error"))

		err := s.client.Grant(ctx, roleName, objectType, objectName)
		s.Error(err)
	})

	s.Run("status error", func() {
		ctx, cancel := context.WithCancel(ctx)
		defer cancel()
		s.mock.ExpectedCalls = nil
		s.mock.EXPECT().OperatePrivilege(tmock.Anything, tmock.Anything).Return(&common.Status{ErrorCode: common.ErrorCode_UnexpectedError}, nil)

		err := s.client.Grant(ctx, roleName, objectType, objectName)
		s.Error(err)
	})

	s.Run("service not ready", func() {
		ctx, cancel := context.WithCancel(ctx)
		defer cancel()

		c := &GrpcClient{}
		err := c.Grant(ctx, roleName, objectType, objectName)
		s.Error(err)
		s.ErrorIs(err, ErrClientNotReady)
	})
}

func (s *RBACSuite) TestRevoke() {
	ctx := context.Background()

	roleName := "testRole"
	objectName := testCollectionName
	objectType := entity.PriviledegeObjectTypeCollection

	s.Run("normal run", func() {
		ctx, cancel := context.WithCancel(ctx)
		defer cancel()
		s.mock.ExpectedCalls = nil
		s.mock.EXPECT().OperatePrivilege(tmock.Anything, tmock.Anything).Run(func(ctx context.Context, req *server.OperatePrivilegeRequest) {
			s.Equal(roleName, req.GetEntity().GetRole().GetName())
			s.Equal(objectName, req.GetEntity().GetObjectName())
			s.Equal(common.ObjectType_name[int32(objectType)], req.GetEntity().GetObject().GetName())
			s.Equal(server.OperatePrivilegeType_Revoke, req.GetType())
		}).Return(&common.Status{ErrorCode: common.ErrorCode_Success}, nil)

		err := s.client.Revoke(ctx, roleName, objectType, objectName)

		s.NoError(err)
	})

	s.Run("rpc error", func() {
		ctx, cancel := context.WithCancel(ctx)
		defer cancel()
		s.mock.ExpectedCalls = nil
		s.mock.EXPECT().OperatePrivilege(tmock.Anything, tmock.Anything).Return(nil, errors.New("mock error"))

		err := s.client.Revoke(ctx, roleName, objectType, objectName)
		s.Error(err)
	})

	s.Run("status error", func() {
		ctx, cancel := context.WithCancel(ctx)
		defer cancel()
		s.mock.ExpectedCalls = nil
		s.mock.EXPECT().OperatePrivilege(tmock.Anything, tmock.Anything).Return(&common.Status{ErrorCode: common.ErrorCode_UnexpectedError}, nil)

		err := s.client.Revoke(ctx, roleName, objectType, objectName)
		s.Error(err)
	})

	s.Run("service not ready", func() {
		ctx, cancel := context.WithCancel(ctx)
		defer cancel()

		c := &GrpcClient{}
		err := c.Revoke(ctx, roleName, objectType, objectName)
		s.Error(err)
		s.ErrorIs(err, ErrClientNotReady)
	})
}

func TestRBACSuite(t *testing.T) {
	suite.Run(t, new(RBACSuite))
}
