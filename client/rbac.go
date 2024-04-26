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

// Grant adds object privileged for role.
func (c *GrpcClient) Grant(ctx context.Context, role string, objectType entity.PriviledgeObjectType, object string) error {
	if c.Service == nil {
		return ErrClientNotReady
	}

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

	resp, err := c.Service.OperatePrivilege(ctx, req)
	if err != nil {
		return err
	}

	return handleRespStatus(resp)
}
