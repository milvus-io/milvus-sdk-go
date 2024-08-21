// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package client

import (
	"context"
	"errors"

	"github.com/milvus-io/milvus-proto/go-api/v2/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v2/milvuspb"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
)

// CreateRole creates a role entity in Milvus.
func (c *GrpcClient) CreateRole(ctx context.Context, name string) error {
	if c.Service == nil {
		return ErrClientNotReady
	}

	req := &milvuspb.CreateRoleRequest{
		Entity: &milvuspb.RoleEntity{
			Name: name,
		},
	}
	resp, err := c.Service.CreateRole(ctx, req)
	if err != nil {
		return err
	}

	return handleRespStatus(resp)
}

// DropRole drops a role entity in Milvus.
func (c *GrpcClient) DropRole(ctx context.Context, name string) error {
	if c.Service == nil {
		return ErrClientNotReady
	}

	req := &milvuspb.DropRoleRequest{
		RoleName: name,
	}

	resp, err := c.Service.DropRole(ctx, req)
	if err != nil {
		return err
	}

	return handleRespStatus(resp)
}

// AddUserRole adds one role for user.
func (c *GrpcClient) AddUserRole(ctx context.Context, username string, role string) error {
	if c.Service == nil {
		return ErrClientNotReady
	}

	req := &milvuspb.OperateUserRoleRequest{
		Username: username,
		RoleName: role,
		Type:     milvuspb.OperateUserRoleType_AddUserToRole,
	}

	resp, err := c.Service.OperateUserRole(ctx, req)
	if err != nil {
		return err
	}

	return handleRespStatus(resp)
}

// RemoveUserRole removes one role from user.
func (c *GrpcClient) RemoveUserRole(ctx context.Context, username string, role string) error {
	if c.Service == nil {
		return ErrClientNotReady
	}

	req := &milvuspb.OperateUserRoleRequest{
		Username: username,
		RoleName: role,
		Type:     milvuspb.OperateUserRoleType_RemoveUserFromRole,
	}

	resp, err := c.Service.OperateUserRole(ctx, req)
	if err != nil {
		return err
	}

	return handleRespStatus(resp)
}

// ListRoles lists the role objects in system.
func (c *GrpcClient) ListRoles(ctx context.Context) ([]entity.Role, error) {
	if c.Service == nil {
		return nil, ErrClientNotReady
	}

	req := &milvuspb.SelectRoleRequest{}

	resp, err := c.Service.SelectRole(ctx, req)
	if err != nil {
		return nil, err
	}
	if err = handleRespStatus(resp.GetStatus()); err != nil {
		return nil, err
	}

	roles := make([]entity.Role, 0, len(resp.GetResults()))
	for _, result := range resp.GetResults() {
		roles = append(roles, entity.Role{Name: result.GetRole().GetName()})
	}

	return roles, nil
}

// ListUsers lists the user objects in system.
func (c *GrpcClient) ListUsers(ctx context.Context) ([]entity.User, error) {
	if c.Service == nil {
		return nil, ErrClientNotReady
	}

	req := &milvuspb.SelectUserRequest{}

	resp, err := c.Service.SelectUser(ctx, req)
	if err != nil {
		return nil, err
	}
	if err = handleRespStatus(resp.GetStatus()); err != nil {
		return nil, err
	}

	users := make([]entity.User, 0, len(resp.GetResults()))
	for _, result := range resp.GetResults() {
		users = append(users, entity.User{Name: result.GetUser().GetName()})
	}

	return users, nil
}

// DescribeUser lists the user descriptions in the system (name, roles)
func (c *GrpcClient) DescribeUser(ctx context.Context, username string) (entity.UserDescription, error) {
	if c.Service == nil {
		return entity.UserDescription{}, ErrClientNotReady
	}

	req := &milvuspb.SelectUserRequest{
		User: &milvuspb.UserEntity{
			Name: username,
		},
		IncludeRoleInfo: true,
	}

	resp, err := c.Service.SelectUser(ctx, req)
	if err != nil {
		return entity.UserDescription{}, err
	}
	if err = handleRespStatus(resp.GetStatus()); err != nil {
		return entity.UserDescription{}, err
	}
	results := resp.GetResults()

	if len(results) == 0 {
		return entity.UserDescription{}, nil
	}

	userDescription := entity.UserDescription{
		Name:  results[0].GetUser().GetName(),
		Roles: make([]string, 0, len(results[0].GetRoles())),
	}

	for _, role := range results[0].GetRoles() {
		userDescription.Roles = append(userDescription.Roles, role.GetName())
	}
	return userDescription, nil
}

// DescribeUsers lists all users with descriptions (names, roles)
func (c *GrpcClient) DescribeUsers(ctx context.Context) ([]entity.UserDescription, error) {
	if c.Service == nil {
		return nil, ErrClientNotReady
	}

	req := &milvuspb.SelectUserRequest{
		IncludeRoleInfo: true,
	}

	resp, err := c.Service.SelectUser(ctx, req)
	if err != nil {
		return nil, err
	}
	if err = handleRespStatus(resp.GetStatus()); err != nil {
		return nil, err
	}
	results := resp.GetResults()

	userDescriptions := make([]entity.UserDescription, 0, len(results))

	for _, result := range results {
		userDescription := entity.UserDescription{
			Name:  result.GetUser().GetName(),
			Roles: make([]string, 0, len(result.GetRoles())),
		}
		for _, role := range result.GetRoles() {
			userDescription.Roles = append(userDescription.Roles, role.GetName())
		}
		userDescriptions = append(userDescriptions, userDescription)
	}

	return userDescriptions, nil
}

// ListGrants lists the role grants in the system
func (c *GrpcClient) ListGrants(ctx context.Context, role string, dbName string) ([]entity.RoleGrants, error) {
	RoleGrantsList := make([]entity.RoleGrants, 0)
	if c.Service == nil {
		return RoleGrantsList, ErrClientNotReady
	}

	req := &milvuspb.SelectGrantRequest{
		Entity: &milvuspb.GrantEntity{
			Role: &milvuspb.RoleEntity{
				Name: role,
			},
			DbName: dbName,
		},
	}

	resp, err := c.Service.SelectGrant(ctx, req)
	if err != nil {
		return RoleGrantsList, err
	}
	if err = handleRespStatus(resp.GetStatus()); err != nil {
		return RoleGrantsList, err
	}

	results := resp.GetEntities()

	if len(results) == 0 {
		return RoleGrantsList, nil
	}

	for _, roleEntity := range results {
		RoleGrant := entity.RoleGrants{
			Object:        roleEntity.Object.Name,
			ObjectName:    roleEntity.ObjectName,
			RoleName:      roleEntity.Role.Name,
			GrantorName:   roleEntity.Grantor.User.Name,
			PrivilegeName: roleEntity.Grantor.Privilege.Name,
			DbName:        roleEntity.DbName,
		}
		RoleGrantsList = append(RoleGrantsList, RoleGrant)
	}

	return RoleGrantsList, nil
}

// ListGrant lists a grant info for the role and the specific object
func (c *GrpcClient) ListGrant(ctx context.Context, role string, object string, objectName string, dbName string) ([]entity.RoleGrants, error) {
	RoleGrantsList := make([]entity.RoleGrants, 0)
	if c.Service == nil {
		return RoleGrantsList, ErrClientNotReady
	}

	req := &milvuspb.SelectGrantRequest{
		Entity: &milvuspb.GrantEntity{
			Role: &milvuspb.RoleEntity{
				Name: role,
			},
			Object: &milvuspb.ObjectEntity{
				Name: object,
			},
			ObjectName: objectName,
			DbName:     dbName,
		},
	}

	resp, err := c.Service.SelectGrant(ctx, req)
	if err != nil {
		return RoleGrantsList, err
	}
	if err = handleRespStatus(resp.GetStatus()); err != nil {
		return RoleGrantsList, err
	}

	results := resp.GetEntities()

	if len(results) == 0 {
		return RoleGrantsList, nil
	}

	for _, roleEntity := range results {
		RoleGrant := entity.RoleGrants{
			Object:        roleEntity.Object.Name,
			ObjectName:    roleEntity.ObjectName,
			RoleName:      roleEntity.Role.Name,
			GrantorName:   roleEntity.Grantor.User.Name,
			PrivilegeName: roleEntity.Grantor.Privilege.Name,
			DbName:        roleEntity.DbName,
		}
		RoleGrantsList = append(RoleGrantsList, RoleGrant)
	}

	return RoleGrantsList, nil
}

// Grant adds object privileged for role.
func (c *GrpcClient) Grant(ctx context.Context, role string, objectType entity.PriviledgeObjectType, object string, privilege string, options ...entity.OperatePrivilegeOption) error {
	if c.Service == nil {
		return ErrClientNotReady
	}

	grantOpt := &entity.OperatePrivilegeOpt{}
	for _, opt := range options {
		opt(grantOpt)
	}

	req := &milvuspb.OperatePrivilegeRequest{
		Base: grantOpt.Base,
		Entity: &milvuspb.GrantEntity{
			Role: &milvuspb.RoleEntity{
				Name: role,
			},
			Object: &milvuspb.ObjectEntity{
				Name: commonpb.ObjectType_name[int32(objectType)],
			},
			Grantor: &milvuspb.GrantorEntity{
				Privilege: &milvuspb.PrivilegeEntity{
					Name: privilege,
				},
			},
			ObjectName: object,
			DbName:     grantOpt.Database,
		},
		Type: milvuspb.OperatePrivilegeType_Grant,
	}

	resp, err := c.Service.OperatePrivilege(ctx, req)
	if err != nil {
		return err
	}

	return handleRespStatus(resp)
}

// Revoke removes privilege from role.
func (c *GrpcClient) Revoke(ctx context.Context, role string, objectType entity.PriviledgeObjectType, object string, privilege string, options ...entity.OperatePrivilegeOption) error {
	if c.Service == nil {
		return ErrClientNotReady
	}
	revokeOpt := &entity.OperatePrivilegeOpt{}
	for _, opt := range options {
		opt(revokeOpt)
	}

	req := &milvuspb.OperatePrivilegeRequest{
		Base: revokeOpt.Base,
		Entity: &milvuspb.GrantEntity{
			Role: &milvuspb.RoleEntity{
				Name: role,
			},
			Object: &milvuspb.ObjectEntity{
				Name: commonpb.ObjectType_name[int32(objectType)],
			},
			ObjectName: object,
			Grantor: &milvuspb.GrantorEntity{
				Privilege: &milvuspb.PrivilegeEntity{
					Name: privilege,
				},
			},
			DbName: revokeOpt.Database,
		},
		Type: milvuspb.OperatePrivilegeType_Revoke,
	}

	resp, err := c.Service.OperatePrivilege(ctx, req)
	if err != nil {
		return err
	}

	return handleRespStatus(resp)
}

func (c *GrpcClient) BackupRBAC(ctx context.Context) (*entity.RBACMeta, error) {
	if c.Service == nil {
		return nil, ErrClientNotReady
	}

	req := &milvuspb.BackupRBACMetaRequest{}

	resp, err := c.Service.BackupRBAC(ctx, req)
	if err != nil {
		return nil, err
	}
	if err = handleRespStatus(resp.GetStatus()); err != nil {
		return nil, err
	}

	meta := resp.GetRBACMeta()
	users := make([]*entity.UserInfo, 0, len(meta.GetUsers()))
	for _, user := range meta.GetUsers() {
		roles := make([]string, 0, len(user.GetRoles()))
		for _, role := range user.GetRoles() {
			roles = append(roles, role.GetName())
		}

		users = append(users, &entity.UserInfo{
			UserDescription: entity.UserDescription{
				Name:  user.GetUser(),
				Roles: roles,
			},
			Password: user.GetPassword(),
		})
	}

	roles := []*entity.Role{}
	for _, role := range meta.GetRoles() {
		roles = append(roles, &entity.Role{
			Name: role.GetName(),
		})
	}

	roleGrants := []*entity.RoleGrants{}
	for _, grant := range meta.GetGrants() {
		roleGrant := &entity.RoleGrants{
			Object:        grant.GetObject().GetName(),
			ObjectName:    grant.GetObjectName(),
			RoleName:      grant.GetRole().GetName(),
			GrantorName:   grant.GetGrantor().GetUser().GetName(),
			PrivilegeName: grant.GetGrantor().GetPrivilege().GetName(),
			DbName:        grant.GetDbName(),
		}

		roleGrants = append(roleGrants, roleGrant)
	}

	return &entity.RBACMeta{
		Users:      users,
		Roles:      roles,
		RoleGrants: roleGrants,
	}, nil
}

func (c *GrpcClient) RestoreRBAC(ctx context.Context, meta *entity.RBACMeta) error {
	if c.Service == nil {
		return ErrClientNotReady
	}

	if meta == nil {
		return errors.New("failed to restore rbac meta, meta is nil")
	}

	users := make([]*milvuspb.UserInfo, 0, len(meta.Users))
	for _, user := range meta.Users {
		roles := make([]*milvuspb.RoleEntity, 0, len(user.Roles))
		for _, role := range meta.Roles {
			roles = append(roles, &milvuspb.RoleEntity{
				Name: role.Name,
			})
		}
		users = append(users, &milvuspb.UserInfo{
			User:     user.Name,
			Password: user.Password,
			Roles:    roles,
		})
	}

	roles := make([]*milvuspb.RoleEntity, 0, len(meta.Roles))
	for _, role := range meta.Roles {
		roles = append(roles, &milvuspb.RoleEntity{
			Name: role.Name,
		})
	}

	grants := make([]*milvuspb.GrantEntity, 0, len(meta.RoleGrants))
	for _, grant := range meta.RoleGrants {
		grants = append(grants, &milvuspb.GrantEntity{
			Role: &milvuspb.RoleEntity{
				Name: grant.RoleName,
			},
			Object: &milvuspb.ObjectEntity{
				Name: grant.Object,
			},
			ObjectName: grant.ObjectName,
			Grantor: &milvuspb.GrantorEntity{
				User: &milvuspb.UserEntity{
					Name: grant.GrantorName,
				},
				Privilege: &milvuspb.PrivilegeEntity{
					Name: grant.PrivilegeName,
				},
			},
			DbName: grant.DbName,
		})
	}

	req := &milvuspb.RestoreRBACMetaRequest{
		RBACMeta: &milvuspb.RBACMeta{
			Users:  users,
			Roles:  roles,
			Grants: grants,
		},
	}

	resp, err := c.Service.RestoreRBAC(ctx, req)
	if err != nil {
		return err
	}

	return handleRespStatus(resp)
}
