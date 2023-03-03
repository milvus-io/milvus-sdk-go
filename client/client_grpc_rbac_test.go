package client

import (
	"context"
	"testing"

	"github.com/cockroachdb/errors"

	"github.com/milvus-io/milvus-sdk-go/v2/entity"
	"github.com/stretchr/testify/suite"

	"github.com/stretchr/testify/mock"

	common "github.com/milvus-io/milvus-proto/go-api/commonpb"
	server "github.com/milvus-io/milvus-proto/go-api/milvuspb"
)

type RBACSuite struct {
	MockSuiteBase
}

func (s *RBACSuite) TestCreateRole() {
	ctx := context.Background()
	roleName := "testRole"

	s.Run("normal run", func() {
		ctx, cancel := context.WithCancel(ctx)
		defer cancel()
		s.mock.ExpectedCalls = nil
		s.mock.EXPECT().CreateRole(mock.Anything, mock.Anything).Run(func(ctx context.Context, req *server.CreateRoleRequest) {
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
		s.mock.EXPECT().CreateRole(mock.Anything, mock.Anything).Return(nil, errors.New("mock error"))
		err := s.client.CreateRole(ctx, roleName)

		s.Error(err)
	})

	s.Run("status error", func() {
		ctx, cancel := context.WithCancel(ctx)
		defer cancel()
		s.mock.ExpectedCalls = nil
		s.mock.EXPECT().CreateRole(mock.Anything, mock.Anything).Return(&common.Status{
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
		s.mock.EXPECT().DropRole(mock.Anything, mock.Anything).Run(func(ctx context.Context, req *server.DropRoleRequest) {
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
		s.mock.EXPECT().DropRole(mock.Anything, mock.Anything).Return(nil, errors.New("mock error"))
		err := s.client.DropRole(ctx, roleName)

		s.Error(err)
	})

	s.Run("status error", func() {
		ctx, cancel := context.WithCancel(ctx)
		defer cancel()
		s.mock.ExpectedCalls = nil
		s.mock.EXPECT().DropRole(mock.Anything, mock.Anything).Return(&common.Status{
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
		s.mock.EXPECT().OperateUserRole(mock.Anything, mock.Anything).Run(func(ctx context.Context, req *server.OperateUserRoleRequest) {
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
		s.mock.EXPECT().OperateUserRole(mock.Anything, mock.Anything).Return(nil, errors.New("mock error"))
		err := s.client.AddUserRole(ctx, username, roleName)

		s.Error(err)
	})

	s.Run("status error", func() {
		ctx, cancel := context.WithCancel(ctx)
		defer cancel()
		s.mock.ExpectedCalls = nil
		s.mock.EXPECT().OperateUserRole(mock.Anything, mock.Anything).Return(&common.Status{
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
		s.mock.EXPECT().OperateUserRole(mock.Anything, mock.Anything).Run(func(ctx context.Context, req *server.OperateUserRoleRequest) {
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
		s.mock.EXPECT().OperateUserRole(mock.Anything, mock.Anything).Return(nil, errors.New("mock error"))
		err := s.client.RemoveUserRole(ctx, username, roleName)

		s.Error(err)
	})

	s.Run("status error", func() {
		ctx, cancel := context.WithCancel(ctx)
		defer cancel()
		s.mock.ExpectedCalls = nil
		s.mock.EXPECT().OperateUserRole(mock.Anything, mock.Anything).Return(&common.Status{
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
		s.mock.EXPECT().SelectRole(mock.Anything, mock.Anything).Run(func(ctx context.Context, req *server.SelectRoleRequest) {
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
		s.mock.EXPECT().SelectRole(mock.Anything, mock.Anything).Return(nil, errors.New("mock error"))

		_, err := s.client.ListRoles(ctx)
		s.Error(err)
	})

	s.Run("status error", func() {
		ctx, cancel := context.WithCancel(ctx)
		defer cancel()
		s.mock.ExpectedCalls = nil
		s.mock.EXPECT().SelectRole(mock.Anything, mock.Anything).Return(&server.SelectRoleResponse{
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
		s.mock.EXPECT().SelectUser(mock.Anything, mock.Anything).Run(func(ctx context.Context, req *server.SelectUserRequest) {
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
		s.mock.EXPECT().SelectUser(mock.Anything, mock.Anything).Return(nil, errors.New("mock error"))

		_, err := s.client.ListUsers(ctx)
		s.Error(err)
	})

	s.Run("status error", func() {
		ctx, cancel := context.WithCancel(ctx)
		defer cancel()
		s.mock.ExpectedCalls = nil
		s.mock.EXPECT().SelectUser(mock.Anything, mock.Anything).Return(&server.SelectUserResponse{
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
		s.mock.EXPECT().OperatePrivilege(mock.Anything, mock.Anything).Run(func(ctx context.Context, req *server.OperatePrivilegeRequest) {
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
		s.mock.EXPECT().OperatePrivilege(mock.Anything, mock.Anything).Return(nil, errors.New("mock error"))

		err := s.client.Grant(ctx, roleName, objectType, objectName)
		s.Error(err)
	})

	s.Run("status error", func() {
		ctx, cancel := context.WithCancel(ctx)
		defer cancel()
		s.mock.ExpectedCalls = nil
		s.mock.EXPECT().OperatePrivilege(mock.Anything, mock.Anything).Return(&common.Status{ErrorCode: common.ErrorCode_UnexpectedError}, nil)

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
		s.mock.EXPECT().OperatePrivilege(mock.Anything, mock.Anything).Run(func(ctx context.Context, req *server.OperatePrivilegeRequest) {
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
		s.mock.EXPECT().OperatePrivilege(mock.Anything, mock.Anything).Return(nil, errors.New("mock error"))

		err := s.client.Revoke(ctx, roleName, objectType, objectName)
		s.Error(err)
	})

	s.Run("status error", func() {
		ctx, cancel := context.WithCancel(ctx)
		defer cancel()
		s.mock.ExpectedCalls = nil
		s.mock.EXPECT().OperatePrivilege(mock.Anything, mock.Anything).Return(&common.Status{ErrorCode: common.ErrorCode_UnexpectedError}, nil)

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
