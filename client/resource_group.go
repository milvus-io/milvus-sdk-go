// Copyright (C) 2019-2021 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License.

package client

import (
	"context"

	"github.com/milvus-io/milvus-proto/go-api/v2/milvuspb"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
)

// ListResourceGroups returns list of resource group names in current Milvus instance.
func (c *GrpcClient) ListResourceGroups(ctx context.Context) ([]string, error) {
	service := c.Service(ctx)
	if service == nil {
		return nil, ErrClientNotReady
	}
	defer service.Close()

	req := &milvuspb.ListResourceGroupsRequest{}

	resp, err := service.ListResourceGroups(ctx, req)
	if err != nil {
		return nil, err
	}
	if err = handleRespStatus(resp.GetStatus()); err != nil {
		return nil, err
	}

	return resp.GetResourceGroups(), nil
}

// CreateResourceGroup creates a resource group with provided name.
func (c *GrpcClient) CreateResourceGroup(ctx context.Context, rgName string) error {
	service := c.Service(ctx)
	if service == nil {
		return ErrClientNotReady
	}
	defer service.Close()

	req := &milvuspb.CreateResourceGroupRequest{
		ResourceGroup: rgName,
	}

	resp, err := service.CreateResourceGroup(ctx, req)
	if err != nil {
		return err
	}
	return handleRespStatus(resp)
}

// DescribeResourceGroup returns resource groups information.
func (c *GrpcClient) DescribeResourceGroup(ctx context.Context, rgName string) (*entity.ResourceGroup, error) {
	service := c.Service(ctx)
	if service == nil {
		return nil, ErrClientNotReady
	}
	defer service.Close()

	req := &milvuspb.DescribeResourceGroupRequest{
		ResourceGroup: rgName,
	}

	resp, err := service.DescribeResourceGroup(ctx, req)
	if err != nil {
		return nil, err
	}
	if err = handleRespStatus(resp.GetStatus()); err != nil {
		return nil, err
	}

	rg := resp.GetResourceGroup()
	result := &entity.ResourceGroup{
		Name:                 rg.GetName(),
		Capacity:             rg.GetCapacity(),
		AvailableNodesNumber: rg.GetNumAvailableNode(),
		LoadedReplica:        rg.GetNumLoadedReplica(),
		OutgoingNodeNum:      rg.GetNumOutgoingNode(),
		IncomingNodeNum:      rg.GetNumIncomingNode(),
	}

	return result, nil
}

// DropResourceGroup drops the resource group with provided name.
func (c *GrpcClient) DropResourceGroup(ctx context.Context, rgName string) error {
	service := c.Service(ctx)
	if service == nil {
		return ErrClientNotReady
	}
	defer service.Close()

	req := &milvuspb.DropResourceGroupRequest{
		ResourceGroup: rgName,
	}

	resp, err := service.DropResourceGroup(ctx, req)
	if err != nil {
		return err
	}
	return handleRespStatus(resp)
}

// TransferNode transfers querynodes between resource groups.
func (c *GrpcClient) TransferNode(ctx context.Context, sourceRg, targetRg string, nodesNum int32) error {
	service := c.Service(ctx)

	if service == nil {
		return ErrClientNotReady
	}

	defer service.Close()
	req := &milvuspb.TransferNodeRequest{
		SourceResourceGroup: sourceRg,
		TargetResourceGroup: targetRg,
		NumNode:             nodesNum,
	}

	resp, err := service.TransferNode(ctx, req)
	if err != nil {
		return err
	}
	return handleRespStatus(resp)
}

// TransferReplica transfer collection replicas between source,target resource group.
func (c *GrpcClient) TransferReplica(ctx context.Context, sourceRg, targetRg string, collectionName string, replicaNum int64) error {
	service := c.Service(ctx)
	if service == nil {
		return ErrClientNotReady
	}
	defer service.Close()

	req := &milvuspb.TransferReplicaRequest{
		SourceResourceGroup: sourceRg,
		TargetResourceGroup: targetRg,
		CollectionName:      collectionName,
		NumReplica:          replicaNum,
	}

	resp, err := service.TransferReplica(ctx, req)
	if err != nil {
		return err
	}
	return handleRespStatus(resp)
}
