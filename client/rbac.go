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

	"github.com/milvus-io/milvus-proto/go-api/v2/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v2/milvuspb"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
)

// CreateRole creates a role entity in Milvus.
func (c *GrpcClient) CreateRole(ctx context.Context, name string) error {
	service := c.Service(ctx)
	if service == nil {
		return ErrClientNotReady
	}
	defer service.Close()

	req := &milvuspb.CreateRoleRequest{
		Entity: &milvuspb.RoleEntity{
			Name: name,
		},
	}
	resp, err := service.CreateRole(ctx, req)
	if err != nil {
		return err
	}

	return handleRespStatus(resp)
}

// DropRole drops a role entity in Milvus.
func (c *GrpcClient) DropRole(ctx context.Context, name string) error {
	service := c.Service(ctx)
	if service == nil {
		return ErrClientNotReady
	}
	defer service.Close()

	req := &milvuspb.DropRoleRequest{
		RoleName: name,
	}

	resp, err := service.DropRole(ctx, req)
	if err != nil {
		return err
	}

	return handleRespStatus(resp)
}

// AddUserRole adds one role for user.
func (c *GrpcClient) AddUserRole(ctx context.Context, username string, role string) error {
	service := c.Service(ctx)
	if service == nil {
		return ErrClientNotReady
	}
	defer service.Close()

	req := &milvuspb.OperateUserRoleRequest{
		Username: username,
		RoleName: role,
		Type:     milvuspb.OperateUserRoleType_AddUserToRole,
	}

	resp, err := service.OperateUserRole(ctx, req)
	if err != nil {
		return err
	}

	return handleRespStatus(resp)
}

// RemoveUserRole removes one role from user.
func (c *GrpcClient) RemoveUserRole(ctx context.Context, username string, role string) error {
	service := c.Service(ctx)
	if service == nil {
		return ErrClientNotReady
	}
	defer service.Close()

	req := &milvuspb.OperateUserRoleRequest{
		Username: username,
		RoleName: role,
		Type:     milvuspb.OperateUserRoleType_RemoveUserFromRole,
	}

	resp, err := service.OperateUserRole(ctx, req)
	if err != nil {
		return err
	}

	return handleRespStatus(resp)
}

// ListRoles lists the role objects in system.
func (c *GrpcClient) ListRoles(ctx context.Context) ([]entity.Role, error) {
	service := c.Service(ctx)
	if service == nil {
		return nil, ErrClientNotReady
	}
	defer service.Close()

	req := &milvuspb.SelectRoleRequest{}

	resp, err := service.SelectRole(ctx, req)
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
	service := c.Service(ctx)
	if service == nil {
		return nil, ErrClientNotReady
	}
	defer service.Close()

	req := &milvuspb.SelectUserRequest{}

	resp, err := service.SelectUser(ctx, req)
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

// Grant adds object privileged for role.
func (c *GrpcClient) Grant(ctx context.Context, role string, objectType entity.PriviledgeObjectType, object string) error {
	service := c.Service(ctx)
	if service == nil {
		return ErrClientNotReady
	}
	defer service.Close()

	req := &milvuspb.OperatePrivilegeRequest{
		Entity: &milvuspb.GrantEntity{
			Role: &milvuspb.RoleEntity{
				Name: role,
			},
			Object: &milvuspb.ObjectEntity{
				Name: commonpb.ObjectType_name[int32(objectType)],
			},
			ObjectName: object,
		},
		Type: milvuspb.OperatePrivilegeType_Grant,
	}

	resp, err := service.OperatePrivilege(ctx, req)
	if err != nil {
		return err
	}

	return handleRespStatus(resp)
}

// Revoke removes privilege from role.
func (c *GrpcClient) Revoke(ctx context.Context, role string, objectType entity.PriviledgeObjectType, object string) error {
	service := c.Service(ctx)
	if service == nil {
		return ErrClientNotReady
	}
	defer service.Close()

	req := &milvuspb.OperatePrivilegeRequest{
		Entity: &milvuspb.GrantEntity{
			Role: &milvuspb.RoleEntity{
				Name: role,
			},
			Object: &milvuspb.ObjectEntity{
				Name: commonpb.ObjectType_name[int32(objectType)],
			},
			ObjectName: object,
		},
		Type: milvuspb.OperatePrivilegeType_Revoke,
	}

	resp, err := service.OperatePrivilege(ctx, req)
	if err != nil {
		return err
	}

	return handleRespStatus(resp)
}
