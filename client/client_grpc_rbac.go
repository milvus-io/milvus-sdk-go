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

	common "github.com/milvus-io/milvus-proto/go-api/commonpb"
	server "github.com/milvus-io/milvus-proto/go-api/milvuspb"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
)

// CreateRole creates a role entity in Milvus.
func (c *GrpcClient) CreateRole(ctx context.Context, name string) error {
	if c.Service == nil {
		return ErrClientNotReady
	}

	req := &server.CreateRoleRequest{
		Entity: &server.RoleEntity{
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

	req := &server.DropRoleRequest{
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

	req := &server.OperateUserRoleRequest{
		Username: username,
		RoleName: role,
		Type:     server.OperateUserRoleType_AddUserToRole,
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

	req := &server.OperateUserRoleRequest{
		Username: username,
		RoleName: role,
		Type:     server.OperateUserRoleType_RemoveUserFromRole,
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

	req := &server.SelectRoleRequest{}

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

	req := &server.SelectUserRequest{}

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

// Grant adds object privileged for role.
func (c *GrpcClient) Grant(ctx context.Context, role string, objectType entity.PriviledgeObjectType, object string) error {
	if c.Service == nil {
		return ErrClientNotReady
	}

	req := &server.OperatePrivilegeRequest{
		Entity: &server.GrantEntity{
			Role: &server.RoleEntity{
				Name: role,
			},
			Object: &server.ObjectEntity{
				Name: common.ObjectType_name[int32(objectType)],
			},
			ObjectName: object,
		},
		Type: server.OperatePrivilegeType_Grant,
	}

	resp, err := c.Service.OperatePrivilege(ctx, req)
	if err != nil {
		return err
	}

	return handleRespStatus(resp)
}

// Revoke removes privilege from role.
func (c *GrpcClient) Revoke(ctx context.Context, role string, objectType entity.PriviledgeObjectType, object string) error {
	if c.Service == nil {
		return ErrClientNotReady
	}

	req := &server.OperatePrivilegeRequest{
		Entity: &server.GrantEntity{
			Role: &server.RoleEntity{
				Name: role,
			},
			Object: &server.ObjectEntity{
				Name: common.ObjectType_name[int32(objectType)],
			},
			ObjectName: object,
		},
		Type: server.OperatePrivilegeType_Revoke,
	}

	resp, err := c.Service.OperatePrivilege(ctx, req)
	if err != nil {
		return err
	}

	return handleRespStatus(resp)
}
