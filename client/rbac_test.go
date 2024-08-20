package client

import (
	"context"
	"testing"

	"github.com/cockroachdb/errors"

	"github.com/milvus-io/milvus-sdk-go/v2/entity"
	"github.com/milvus-io/milvus-sdk-go/v2/merr"
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

func (s *RBACSuite) TestDescribeUser() {
	ctx := context.Background()
	userName := "testUser"

	s.Run("normal run", func() {
		ctx, cancel := context.WithCancel(ctx)
		defer cancel()
		defer s.resetMock()
		s.mock.EXPECT().SelectUser(mock.Anything, mock.Anything).Run(func(ctx context.Context, req *milvuspb.SelectUserRequest) {
			s.True(req.GetIncludeRoleInfo())
			s.Equal(req.GetUser().GetName(), userName)
		}).Return(&milvuspb.SelectUserResponse{
			Status: &commonpb.Status{ErrorCode: commonpb.ErrorCode_Success},
			Results: []*milvuspb.UserResult{
				{
					User: &milvuspb.UserEntity{
						Name: userName,
					},
					Roles: []*milvuspb.RoleEntity{
						{Name: "role1"},
						{Name: "role2"},
					},
				},
			},
		}, nil)

		userDesc, err := s.client.DescribeUser(ctx, userName)

		s.NoError(err)
		s.Equal(userDesc.Name, userName)
		s.ElementsMatch(userDesc.Roles, []string{"role1", "role2"})
	})

	s.Run("rpc error", func() {
		ctx, cancel := context.WithCancel(ctx)
		defer cancel()
		defer s.resetMock()
		s.mock.EXPECT().SelectUser(mock.Anything, mock.Anything).Return(nil, errors.New("mock error"))

		_, err := s.client.DescribeUser(ctx, userName)
		s.Error(err)
	})

	s.Run("status error", func() {
		ctx, cancel := context.WithCancel(ctx)
		defer cancel()
		defer s.resetMock()
		s.mock.EXPECT().SelectUser(mock.Anything, mock.Anything).Return(&milvuspb.SelectUserResponse{
			Status: &commonpb.Status{ErrorCode: commonpb.ErrorCode_UnexpectedError},
		}, nil)

		_, err := s.client.DescribeUser(ctx, userName)
		s.Error(err)
	})

	s.Run("service not ready", func() {
		ctx, cancel := context.WithCancel(ctx)
		defer cancel()

		c := &GrpcClient{}
		_, err := c.DescribeUser(ctx, userName)
		s.Error(err)
		s.ErrorIs(err, ErrClientNotReady)
	})
}

func (s *RBACSuite) TestDescribeUsers() {
	ctx := context.Background()

	s.Run("normal run", func() {
		ctx, cancel := context.WithCancel(ctx)
		defer cancel()
		defer s.resetMock()

		mockResults := []*milvuspb.UserResult{
			{
				User: &milvuspb.UserEntity{
					Name: "user1",
				},
				Roles: []*milvuspb.RoleEntity{
					{Name: "role1"},
					{Name: "role2"},
				},
			},
			{
				User: &milvuspb.UserEntity{
					Name: "user2",
				},
				Roles: []*milvuspb.RoleEntity{
					{Name: "role3"},
				},
			},
		}

		s.mock.EXPECT().SelectUser(mock.Anything, mock.Anything).Run(func(ctx context.Context, req *milvuspb.SelectUserRequest) {
			s.True(req.GetIncludeRoleInfo())
		}).Return(&milvuspb.SelectUserResponse{
			Status:  &commonpb.Status{ErrorCode: commonpb.ErrorCode_Success},
			Results: mockResults,
		}, nil)

		userDescs, err := s.client.DescribeUsers(ctx)

		s.NoError(err)
		s.Len(userDescs, 2)

		expectedDescs := []entity.UserDescription{
			{
				Name:  "user1",
				Roles: []string{"role1", "role2"},
			},
			{
				Name:  "user2",
				Roles: []string{"role3"},
			},
		}

		for i, desc := range userDescs {
			s.Equal(expectedDescs[i].Name, desc.Name)
			s.ElementsMatch(expectedDescs[i].Roles, desc.Roles)
		}
	})

	s.Run("rpc error", func() {
		ctx, cancel := context.WithCancel(ctx)
		defer cancel()
		defer s.resetMock()

		s.mock.EXPECT().SelectUser(mock.Anything, mock.Anything).Return(nil, errors.New("mock error"))

		_, err := s.client.DescribeUsers(ctx)
		s.Error(err)
	})

	s.Run("status error", func() {
		ctx, cancel := context.WithCancel(ctx)
		defer cancel()
		defer s.resetMock()

		s.mock.EXPECT().SelectUser(mock.Anything, mock.Anything).Return(&milvuspb.SelectUserResponse{
			Status: &commonpb.Status{ErrorCode: commonpb.ErrorCode_UnexpectedError},
		}, nil)

		_, err := s.client.DescribeUsers(ctx)
		s.Error(err)
	})

	s.Run("service not ready", func() {
		ctx, cancel := context.WithCancel(ctx)
		defer cancel()

		c := &GrpcClient{}
		_, err := c.DescribeUsers(ctx)
		s.Error(err)
		s.ErrorIs(err, ErrClientNotReady)
	})
}

func (s *RBACSuite) TestListGrant() {
	ctx := context.Background()
	roleName := "testRole"
	object := "testObject"
	objectName := "testObjectName"
	dbName := "testDB"

	s.Run("normal run", func() {
		ctx, cancel := context.WithCancel(ctx)
		defer cancel()
		defer s.resetMock()

		mockResults := []*milvuspb.GrantEntity{
			{
				Object: &milvuspb.ObjectEntity{
					Name: object,
				},
				ObjectName: objectName,
				Role: &milvuspb.RoleEntity{
					Name: roleName,
				},
				Grantor: &milvuspb.GrantorEntity{
					User: &milvuspb.UserEntity{
						Name: "grantorUser",
					},
					Privilege: &milvuspb.PrivilegeEntity{
						Name: "testPrivilege",
					},
				},
				DbName: dbName,
			},
		}

		s.mock.EXPECT().SelectGrant(mock.Anything, mock.Anything).Run(func(ctx context.Context, req *milvuspb.SelectGrantRequest) {
			s.Equal(req.GetEntity().GetRole().GetName(), roleName)
			s.Equal(req.GetEntity().GetObject().GetName(), object)
			s.Equal(req.GetEntity().GetObjectName(), objectName)
			s.Equal(req.GetEntity().GetDbName(), dbName)
		}).Return(&milvuspb.SelectGrantResponse{
			Status:   &commonpb.Status{ErrorCode: commonpb.ErrorCode_Success},
			Entities: mockResults,
		}, nil)

		grants, err := s.client.ListGrant(ctx, roleName, object, objectName, dbName)

		s.NoError(err)
		s.Len(grants, 1)

		expectedGrant := entity.RoleGrants{
			Object:        object,
			ObjectName:    objectName,
			RoleName:      roleName,
			GrantorName:   "grantorUser",
			PrivilegeName: "testPrivilege",
			DbName:        dbName,
		}

		s.Equal(expectedGrant, grants[0])
	})
}

func (s *RBACSuite) TestListGrants() {
	ctx := context.Background()
	roleName := "testRole"
	dbName := "testDB"

	s.Run("normal run", func() {
		ctx, cancel := context.WithCancel(ctx)
		defer cancel()
		defer s.resetMock()

		mockResults := []*milvuspb.GrantEntity{
			{
				Object: &milvuspb.ObjectEntity{
					Name: "testObject",
				},
				ObjectName: "testObjectName",
				Role: &milvuspb.RoleEntity{
					Name: roleName,
				},
				Grantor: &milvuspb.GrantorEntity{
					User: &milvuspb.UserEntity{
						Name: "grantorUser",
					},
					Privilege: &milvuspb.PrivilegeEntity{
						Name: "testPrivilege",
					},
				},
				DbName: dbName,
			},
		}

		s.mock.EXPECT().SelectGrant(mock.Anything, mock.Anything).Run(func(ctx context.Context, req *milvuspb.SelectGrantRequest) {
			s.Equal(req.GetEntity().GetRole().GetName(), roleName)
			s.Equal(req.GetEntity().GetDbName(), dbName)
		}).Return(&milvuspb.SelectGrantResponse{
			Status:   &commonpb.Status{ErrorCode: commonpb.ErrorCode_Success},
			Entities: mockResults,
		}, nil)

		grants, err := s.client.ListGrants(ctx, roleName, dbName)

		s.NoError(err)
		s.Len(grants, 1)

		expectedGrant := entity.RoleGrants{
			Object:        "testObject",
			ObjectName:    "testObjectName",
			RoleName:      roleName,
			GrantorName:   "grantorUser",
			PrivilegeName: "testPrivilege",
			DbName:        dbName,
		}

		s.Equal(expectedGrant, grants[0])
	})
}

func (s *RBACSuite) TestGrant() {
	ctx := context.Background()

	roleName := "testRole"
	objectName := testCollectionName
	objectType := entity.PriviledegeObjectTypeCollection
	dbName := "testDB"
	privilege := "testPrivilege"

	s.Run("normal run", func() {
		ctx, cancel := context.WithCancel(ctx)
		defer cancel()
		defer s.resetMock()
		s.mock.EXPECT().OperatePrivilege(mock.Anything, mock.Anything).Run(func(ctx context.Context, req *milvuspb.OperatePrivilegeRequest) {
			s.Equal(roleName, req.GetEntity().GetRole().GetName())
			s.Equal(objectName, req.GetEntity().GetObjectName())
			s.Equal(commonpb.ObjectType_name[int32(objectType)], req.GetEntity().GetObject().GetName())
			s.Equal(milvuspb.OperatePrivilegeType_Grant, req.GetType())
			s.Equal(privilege, req.GetEntity().GetGrantor().GetPrivilege().GetName())
			s.Equal(dbName, req.GetEntity().GetDbName())
		}).Return(&commonpb.Status{ErrorCode: commonpb.ErrorCode_Success}, nil)

		err := s.client.Grant(ctx, roleName, objectType, objectName, privilege, entity.WithOperatePrivilegeDatabase(dbName))

		s.NoError(err)
	})

	s.Run("rpc error", func() {
		ctx, cancel := context.WithCancel(ctx)
		defer cancel()
		defer s.resetMock()
		s.mock.EXPECT().OperatePrivilege(mock.Anything, mock.Anything).Return(nil, errors.New("mock error"))

		err := s.client.Grant(ctx, roleName, objectType, objectName, privilege, entity.WithOperatePrivilegeDatabase(dbName))
		s.Error(err)
	})

	s.Run("status error", func() {
		ctx, cancel := context.WithCancel(ctx)
		defer cancel()
		defer s.resetMock()
		s.mock.EXPECT().OperatePrivilege(mock.Anything, mock.Anything).Return(&commonpb.Status{ErrorCode: commonpb.ErrorCode_UnexpectedError}, nil)

		err := s.client.Grant(ctx, roleName, objectType, objectName, privilege, entity.WithOperatePrivilegeDatabase(dbName))
		s.Error(err)
	})

	s.Run("service not ready", func() {
		ctx, cancel := context.WithCancel(ctx)
		defer cancel()

		c := &GrpcClient{}
		err := c.Grant(ctx, roleName, objectType, objectName, privilege, entity.WithOperatePrivilegeDatabase(dbName))
		s.Error(err)
		s.ErrorIs(err, ErrClientNotReady)
	})
}

func (s *RBACSuite) TestRevoke() {
	ctx := context.Background()

	roleName := "testRole"
	objectName := testCollectionName
	objectType := entity.PriviledegeObjectTypeCollection
	dbName := "testDB"
	privilege := "testPrivilege"

	s.Run("normal run", func() {
		ctx, cancel := context.WithCancel(ctx)
		defer cancel()
		defer s.resetMock()
		s.mock.EXPECT().OperatePrivilege(mock.Anything, mock.Anything).Run(func(ctx context.Context, req *milvuspb.OperatePrivilegeRequest) {
			s.Equal(roleName, req.GetEntity().GetRole().GetName())
			s.Equal(objectName, req.GetEntity().GetObjectName())
			s.Equal(commonpb.ObjectType_name[int32(objectType)], req.GetEntity().GetObject().GetName())
			s.Equal(milvuspb.OperatePrivilegeType_Revoke, req.GetType())
			s.Equal(privilege, req.GetEntity().GetGrantor().GetPrivilege().GetName())
			s.Equal(dbName, req.GetEntity().GetDbName())
		}).Return(&commonpb.Status{ErrorCode: commonpb.ErrorCode_Success}, nil)

		err := s.client.Revoke(ctx, roleName, objectType, objectName, privilege, entity.WithOperatePrivilegeDatabase(dbName))

		s.NoError(err)
	})

	s.Run("rpc error", func() {
		ctx, cancel := context.WithCancel(ctx)
		defer cancel()
		defer s.resetMock()
		s.mock.EXPECT().OperatePrivilege(mock.Anything, mock.Anything).Return(nil, errors.New("mock error"))

		err := s.client.Revoke(ctx, roleName, objectType, objectName, privilege, entity.WithOperatePrivilegeDatabase(dbName))
		s.Error(err)
	})

	s.Run("status error", func() {
		ctx, cancel := context.WithCancel(ctx)
		defer cancel()
		defer s.resetMock()
		s.mock.EXPECT().OperatePrivilege(mock.Anything, mock.Anything).Return(&commonpb.Status{ErrorCode: commonpb.ErrorCode_UnexpectedError}, nil)

		err := s.client.Revoke(ctx, roleName, objectType, objectName, privilege, entity.WithOperatePrivilegeDatabase(dbName))
		s.Error(err)
	})

	s.Run("service not ready", func() {
		ctx, cancel := context.WithCancel(ctx)
		defer cancel()
		defer s.resetMock()

		c := &GrpcClient{}
		err := c.Revoke(ctx, roleName, objectType, objectName, privilege, entity.WithOperatePrivilegeDatabase(dbName))
		s.Error(err)
		s.ErrorIs(err, ErrClientNotReady)
	})
}

func (s *RBACSuite) TestBackup() {
	ctx := context.Background()

	s.Run("normal run", func() {
		ctx, cancel := context.WithCancel(ctx)
		defer cancel()
		defer s.resetMock()
		s.mock.EXPECT().BackupRBAC(mock.Anything, mock.Anything).Run(func(_ context.Context, _ *milvuspb.BackupRBACMetaRequest) {
		}).Return(&milvuspb.BackupRBACMetaResponse{
			Status: merr.Success(),
			RBACMeta: &milvuspb.RBACMeta{
				Users: []*milvuspb.UserInfo{
					{
						User:     "user1",
						Password: "passwd",
						Roles: []*milvuspb.RoleEntity{
							{Name: "role1"},
						},
					},
				},
				Roles: []*milvuspb.RoleEntity{
					{Name: "role1"},
				},
				Grants: []*milvuspb.GrantEntity{
					{
						Object: &milvuspb.ObjectEntity{
							Name: "testObject",
						},
						ObjectName: "testObjectName",
						Role: &milvuspb.RoleEntity{
							Name: "testRole",
						},
						Grantor: &milvuspb.GrantorEntity{
							User: &milvuspb.UserEntity{
								Name: "grantorUser",
							},
							Privilege: &milvuspb.PrivilegeEntity{
								Name: "testPrivilege",
							},
						},
						DbName: "testDB",
					},
				},
			},
		}, nil)

		meta, err := s.client.BackupRBAC(ctx)
		s.NoError(err)
		s.Len(meta.Users, 1)
		s.Len(meta.Roles, 1)
		s.Len(meta.RoleGrants, 1)
	})

	s.Run("rpc error", func() {
		ctx, cancel := context.WithCancel(ctx)
		defer cancel()
		defer s.resetMock()
		s.mock.EXPECT().BackupRBAC(mock.Anything, mock.Anything).Return(nil, errors.New("mock error"))

		_, err := s.client.BackupRBAC(ctx)
		s.Error(err)
	})

	s.Run("status error", func() {
		ctx, cancel := context.WithCancel(ctx)
		defer cancel()
		defer s.resetMock()
		s.mock.EXPECT().BackupRBAC(mock.Anything, mock.Anything).Return(&milvuspb.BackupRBACMetaResponse{
			Status: &commonpb.Status{ErrorCode: commonpb.ErrorCode_UnexpectedError},
		}, nil)

		_, err := s.client.BackupRBAC(ctx)
		s.Error(err)
	})

	s.Run("service not ready", func() {
		ctx, cancel := context.WithCancel(ctx)
		defer cancel()
		defer s.resetMock()

		c := &GrpcClient{}
		_, err := c.BackupRBAC(ctx)
		s.Error(err)
		s.ErrorIs(err, ErrClientNotReady)
	})
}

func (s *RBACSuite) TestRestore() {
	ctx := context.Background()

	s.Run("normal run", func() {
		ctx, cancel := context.WithCancel(ctx)
		defer cancel()
		defer s.resetMock()

		s.mock.EXPECT().RestoreRBAC(mock.Anything, mock.Anything).Run(func(_ context.Context, req *milvuspb.RestoreRBACMetaRequest) {
			s.Len(req.RBACMeta.Users, 1)
			s.Len(req.RBACMeta.Roles, 1)
			s.Len(req.RBACMeta.Grants, 1)
		}).Return(merr.Success(), nil)

		err := s.client.RestoreRBAC(ctx, &entity.RBACMeta{
			Users: []*entity.UserInfo{
				{
					UserDescription: entity.UserDescription{
						Name:  "user1",
						Roles: []string{"role1"},
					},
					Password: "passwd",
				},
			},
			Roles: []*entity.Role{
				{Name: "role1"},
			},
			RoleGrants: []*entity.RoleGrants{
				{
					Object:        "testObject",
					ObjectName:    "testObjectName",
					RoleName:      "testRole",
					GrantorName:   "grantorUser",
					PrivilegeName: "testPrivilege",
					DbName:        "testDB",
				},
			},
		})
		s.NoError(err)
	})

	s.Run("rpc error", func() {
		ctx, cancel := context.WithCancel(ctx)
		defer cancel()
		defer s.resetMock()
		s.mock.EXPECT().RestoreRBAC(mock.Anything, mock.Anything).Return(nil, errors.New("mock error"))

		err := s.client.RestoreRBAC(ctx, &entity.RBACMeta{})
		s.Error(err)
	})

	s.Run("status error", func() {
		ctx, cancel := context.WithCancel(ctx)
		defer cancel()
		defer s.resetMock()
		s.mock.EXPECT().RestoreRBAC(mock.Anything, mock.Anything).Return(&commonpb.Status{ErrorCode: commonpb.ErrorCode_UnexpectedError}, nil)

		err := s.client.RestoreRBAC(ctx, &entity.RBACMeta{})
		s.Error(err)
	})

	s.Run("service not ready", func() {
		ctx, cancel := context.WithCancel(ctx)
		defer cancel()
		defer s.resetMock()

		c := &GrpcClient{}
		err := c.RestoreRBAC(ctx, &entity.RBACMeta{})
		s.Error(err)
		s.ErrorIs(err, ErrClientNotReady)
	})
}

func TestRBACSuite(t *testing.T) {
	suite.Run(t, new(RBACSuite))
}
