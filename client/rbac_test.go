package client

import (
	"context"
	"testing"

	"github.com/cockroachdb/errors"

	"github.com/milvus-io/milvus-sdk-go/v2/entity"
	"github.com/stretchr/testify/suite"

	"github.com/stretchr/testify/mock"

	"github.com/milvus-io/milvus-proto/go-api/v2/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v2/milvuspb"
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
		defer s.resetMock()
		s.mock.EXPECT().CreateRole(mock.Anything, mock.Anything).Run(func(ctx context.Context, req *milvuspb.CreateRoleRequest) {
			s.Equal(roleName, req.GetEntity().GetName())
		}).Return(&commonpb.Status{
			ErrorCode: commonpb.ErrorCode_Success,
		}, nil)
		err := s.client.CreateRole(ctx, roleName)

		s.NoError(err)
	})

	s.Run("rpc error", func() {
		ctx, cancel := context.WithCancel(ctx)
		defer cancel()
		defer s.resetMock()
		s.mock.EXPECT().CreateRole(mock.Anything, mock.Anything).Return(nil, errors.New("mock error"))
		err := s.client.CreateRole(ctx, roleName)

		s.Error(err)
	})

	s.Run("status error", func() {
		ctx, cancel := context.WithCancel(ctx)
		defer cancel()
		defer s.resetMock()
		s.mock.EXPECT().CreateRole(mock.Anything, mock.Anything).Return(&commonpb.Status{
			ErrorCode: commonpb.ErrorCode_UnexpectedError,
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
		defer s.resetMock()
		s.mock.EXPECT().DropRole(mock.Anything, mock.Anything).Run(func(ctx context.Context, req *milvuspb.DropRoleRequest) {
			s.Equal(roleName, req.GetRoleName())
		}).Return(&commonpb.Status{
			ErrorCode: commonpb.ErrorCode_Success,
		}, nil)
		err := s.client.DropRole(ctx, roleName)

		s.NoError(err)
	})

	s.Run("rpc error", func() {
		ctx, cancel := context.WithCancel(ctx)
		defer cancel()
		defer s.resetMock()
		s.mock.EXPECT().DropRole(mock.Anything, mock.Anything).Return(nil, errors.New("mock error"))
		err := s.client.DropRole(ctx, roleName)

		s.Error(err)
	})

	s.Run("status error", func() {
		ctx, cancel := context.WithCancel(ctx)
		defer cancel()
		defer s.resetMock()
		s.mock.EXPECT().DropRole(mock.Anything, mock.Anything).Return(&commonpb.Status{
			ErrorCode: commonpb.ErrorCode_UnexpectedError,
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
		defer s.resetMock()
		s.mock.EXPECT().OperateUserRole(mock.Anything, mock.Anything).Run(func(ctx context.Context, req *milvuspb.OperateUserRoleRequest) {
			s.Equal(milvuspb.OperateUserRoleType_AddUserToRole, req.GetType())
			s.Equal(username, req.GetUsername())
			s.Equal(roleName, req.GetRoleName())
		}).Return(&commonpb.Status{
			ErrorCode: commonpb.ErrorCode_Success,
		}, nil)
		err := s.client.AddUserRole(ctx, username, roleName)

		s.NoError(err)
	})

	s.Run("rpc error", func() {
		ctx, cancel := context.WithCancel(ctx)
		defer cancel()
		defer s.resetMock()
		s.mock.EXPECT().OperateUserRole(mock.Anything, mock.Anything).Return(nil, errors.New("mock error"))
		err := s.client.AddUserRole(ctx, username, roleName)

		s.Error(err)
	})

	s.Run("status error", func() {
		ctx, cancel := context.WithCancel(ctx)
		defer cancel()
		defer s.resetMock()
		s.mock.EXPECT().OperateUserRole(mock.Anything, mock.Anything).Return(&commonpb.Status{
			ErrorCode: commonpb.ErrorCode_UnexpectedError,
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
		defer s.resetMock()
		s.mock.EXPECT().OperateUserRole(mock.Anything, mock.Anything).Run(func(ctx context.Context, req *milvuspb.OperateUserRoleRequest) {
			s.Equal(milvuspb.OperateUserRoleType_RemoveUserFromRole, req.GetType())
			s.Equal(username, req.GetUsername())
			s.Equal(roleName, req.GetRoleName())
		}).Return(&commonpb.Status{
			ErrorCode: commonpb.ErrorCode_Success,
		}, nil)
		err := s.client.RemoveUserRole(ctx, username, roleName)

		s.NoError(err)
	})

	s.Run("rpc error", func() {
		ctx, cancel := context.WithCancel(ctx)
		defer cancel()
		defer s.resetMock()
		s.mock.EXPECT().OperateUserRole(mock.Anything, mock.Anything).Return(nil, errors.New("mock error"))
		err := s.client.RemoveUserRole(ctx, username, roleName)

		s.Error(err)
	})

	s.Run("status error", func() {
		ctx, cancel := context.WithCancel(ctx)
		defer cancel()
		defer s.resetMock()
		s.mock.EXPECT().OperateUserRole(mock.Anything, mock.Anything).Return(&commonpb.Status{
			ErrorCode: commonpb.ErrorCode_UnexpectedError,
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
		defer s.resetMock()
		s.mock.EXPECT().SelectRole(mock.Anything, mock.Anything).Run(func(ctx context.Context, req *milvuspb.SelectRoleRequest) {
			s.False(req.GetIncludeUserInfo())
		}).Return(&milvuspb.SelectRoleResponse{
			Status: &commonpb.Status{ErrorCode: commonpb.ErrorCode_Success},
			Results: []*milvuspb.RoleResult{
				{
					Role: &milvuspb.RoleEntity{
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
		defer s.resetMock()
		s.mock.EXPECT().SelectRole(mock.Anything, mock.Anything).Return(nil, errors.New("mock error"))

		_, err := s.client.ListRoles(ctx)
		s.Error(err)
	})

	s.Run("status error", func() {
		ctx, cancel := context.WithCancel(ctx)
		defer cancel()
		defer s.resetMock()
		s.mock.EXPECT().SelectRole(mock.Anything, mock.Anything).Return(&milvuspb.SelectRoleResponse{
			Status: &commonpb.Status{ErrorCode: commonpb.ErrorCode_UnexpectedError},
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
		defer s.resetMock()
		s.mock.EXPECT().SelectUser(mock.Anything, mock.Anything).Run(func(ctx context.Context, req *milvuspb.SelectUserRequest) {
			s.False(req.GetIncludeRoleInfo())
		}).Return(&milvuspb.SelectUserResponse{
			Status: &commonpb.Status{ErrorCode: commonpb.ErrorCode_Success},
			Results: []*milvuspb.UserResult{
				{
					User: &milvuspb.UserEntity{
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
		defer s.resetMock()
		s.mock.EXPECT().SelectUser(mock.Anything, mock.Anything).Return(nil, errors.New("mock error"))

		_, err := s.client.ListUsers(ctx)
		s.Error(err)
	})

	s.Run("status error", func() {
		ctx, cancel := context.WithCancel(ctx)
		defer cancel()
		defer s.resetMock()
		s.mock.EXPECT().SelectUser(mock.Anything, mock.Anything).Return(&milvuspb.SelectUserResponse{
			Status: &commonpb.Status{ErrorCode: commonpb.ErrorCode_UnexpectedError},
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
		defer s.resetMock()
		s.mock.EXPECT().OperatePrivilege(mock.Anything, mock.Anything).Run(func(ctx context.Context, req *milvuspb.OperatePrivilegeRequest) {
			s.Equal(roleName, req.GetEntity().GetRole().GetName())
			s.Equal(objectName, req.GetEntity().GetObjectName())
			s.Equal(commonpb.ObjectType_name[int32(objectType)], req.GetEntity().GetObject().GetName())
			s.Equal(milvuspb.OperatePrivilegeType_Grant, req.GetType())
		}).Return(&commonpb.Status{ErrorCode: commonpb.ErrorCode_Success}, nil)

		err := s.client.Grant(ctx, roleName, objectType, objectName)

		s.NoError(err)
	})

	s.Run("rpc error", func() {
		ctx, cancel := context.WithCancel(ctx)
		defer cancel()
		defer s.resetMock()
		s.mock.EXPECT().OperatePrivilege(mock.Anything, mock.Anything).Return(nil, errors.New("mock error"))

		err := s.client.Grant(ctx, roleName, objectType, objectName)
		s.Error(err)
	})

	s.Run("status error", func() {
		ctx, cancel := context.WithCancel(ctx)
		defer cancel()
		defer s.resetMock()
		s.mock.EXPECT().OperatePrivilege(mock.Anything, mock.Anything).Return(&commonpb.Status{ErrorCode: commonpb.ErrorCode_UnexpectedError}, nil)

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
		defer s.resetMock()
		s.mock.EXPECT().OperatePrivilege(mock.Anything, mock.Anything).Run(func(ctx context.Context, req *milvuspb.OperatePrivilegeRequest) {
			s.Equal(roleName, req.GetEntity().GetRole().GetName())
			s.Equal(objectName, req.GetEntity().GetObjectName())
			s.Equal(commonpb.ObjectType_name[int32(objectType)], req.GetEntity().GetObject().GetName())
			s.Equal(milvuspb.OperatePrivilegeType_Revoke, req.GetType())
		}).Return(&commonpb.Status{ErrorCode: commonpb.ErrorCode_Success}, nil)

		err := s.client.Revoke(ctx, roleName, objectType, objectName)

		s.NoError(err)
	})

	s.Run("rpc error", func() {
		ctx, cancel := context.WithCancel(ctx)
		defer cancel()
		defer s.resetMock()
		s.mock.EXPECT().OperatePrivilege(mock.Anything, mock.Anything).Return(nil, errors.New("mock error"))

		err := s.client.Revoke(ctx, roleName, objectType, objectName)
		s.Error(err)
	})

	s.Run("status error", func() {
		ctx, cancel := context.WithCancel(ctx)
		defer cancel()
		defer s.resetMock()
		s.mock.EXPECT().OperatePrivilege(mock.Anything, mock.Anything).Return(&commonpb.Status{ErrorCode: commonpb.ErrorCode_UnexpectedError}, nil)

		err := s.client.Revoke(ctx, roleName, objectType, objectName)
		s.Error(err)
	})

	s.Run("service not ready", func() {
		ctx, cancel := context.WithCancel(ctx)
		defer cancel()
		defer s.resetMock()

		c := &GrpcClient{}
		err := c.Revoke(ctx, roleName, objectType, objectName)
		s.Error(err)
		s.ErrorIs(err, ErrClientNotReady)
	})
}

func TestRBACSuite(t *testing.T) {
	suite.Run(t, new(RBACSuite))
}
