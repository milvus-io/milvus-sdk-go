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
	"log"

	"go.opentelemetry.io/otel"

	"github.com/milvus-io/milvus-proto/go-api/v2/milvuspb"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
)

// ListResourceGroups returns list of resource group names in current Milvus instance.
func (c *GrpcClient) ListResourceGroups(ctx context.Context) ([]string, error) {
	method := "ListResourceGroups"
	ctx, span := otel.Tracer("client").Start(ctx, method)
	defer span.End()
	traceID := span.SpanContext().TraceID().String()
	if c.Service == nil {
		return nil, ErrClientNotReady
	}

	req := &milvuspb.ListResourceGroupsRequest{}

	resp, err := c.Service.ListResourceGroups(ctx, req)
	if err != nil {
		log.Fatalf("list resource groups failed, traceID:%s, error: %v", traceID, err)
		return nil, err
	}
	if err = handleRespStatus(resp.GetStatus()); err != nil {
		log.Fatalf("list resource groups failed, traceID:%s, error: %v", traceID, err)
		return nil, err
	}

	return resp.GetResourceGroups(), nil
}

// CreateResourceGroup creates a resource group with provided name.
func (c *GrpcClient) CreateResourceGroup(ctx context.Context, rgName string, opts ...CreateResourceGroupOption) error {
	method := "CreateResourceGroup"
	ctx, span := otel.Tracer("client").Start(ctx, method)
	defer span.End()
	traceID := span.SpanContext().TraceID().String()
	if c.Service == nil {
		return ErrClientNotReady
	}

	req := &milvuspb.CreateResourceGroupRequest{
		ResourceGroup: rgName,
	}
	for _, opt := range opts {
		opt(req)
	}

	resp, err := c.Service.CreateResourceGroup(ctx, req)
	if err != nil {
		log.Fatalf("create resource group failed, traceID:%s, error: %v", traceID, err)
		return err
	}
	return handleRespStatus(resp)
}

// UpdateResourceGroups updates resource groups with provided options.
func (c *GrpcClient) UpdateResourceGroups(ctx context.Context, opts ...UpdateResourceGroupsOption) error {
	method := "UpdateResourceGroups"
	ctx, span := otel.Tracer("client").Start(ctx, method)
	defer span.End()
	traceID := span.SpanContext().TraceID().String()
	if c.Service == nil {
		return ErrClientNotReady
	}

	req := &milvuspb.UpdateResourceGroupsRequest{}
	for _, opt := range opts {
		opt(req)
	}

	resp, err := c.Service.UpdateResourceGroups(ctx, req)
	if err != nil {
		log.Fatalf("update resource groups failed, traceID:%s, error: %v", traceID, err)
		return err
	}
	return handleRespStatus(resp)
}

// DescribeResourceGroup returns resource groups information.
func (c *GrpcClient) DescribeResourceGroup(ctx context.Context, rgName string) (*entity.ResourceGroup, error) {
	method := "DescribeResourceGroup"
	ctx, span := otel.Tracer("client").Start(ctx, method)
	defer span.End()
	traceID := span.SpanContext().TraceID().String()
	if c.Service == nil {
		return nil, ErrClientNotReady
	}

	req := &milvuspb.DescribeResourceGroupRequest{
		ResourceGroup: rgName,
	}

	resp, err := c.Service.DescribeResourceGroup(ctx, req)
	if err != nil {
		log.Fatalf("describe resource group failed, traceID:%s, error: %v", traceID, err)
		return nil, err
	}
	if err = handleRespStatus(resp.GetStatus()); err != nil {
		log.Fatalf("describe resource group failed, traceID:%s, error: %v", traceID, err)
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
		Config:               rg.GetConfig(),
		Nodes:                rg.GetNodes(),
	}

	return result, nil
}

// DropResourceGroup drops the resource group with provided name.
func (c *GrpcClient) DropResourceGroup(ctx context.Context, rgName string) error {
	method := "DropResourceGroup"
	ctx, span := otel.Tracer("client").Start(ctx, method)
	defer span.End()
	traceID := span.SpanContext().TraceID().String()
	if c.Service == nil {
		return ErrClientNotReady
	}

	req := &milvuspb.DropResourceGroupRequest{
		ResourceGroup: rgName,
	}

	resp, err := c.Service.DropResourceGroup(ctx, req)
	if err != nil {
		log.Fatalf("drop resource group failed, traceID:%s, error: %v", traceID, err)
		return err
	}
	return handleRespStatus(resp)
}

// TransferNode transfers querynodes between resource groups.
func (c *GrpcClient) TransferNode(ctx context.Context, sourceRg, targetRg string, nodesNum int32) error {
	method := "TransferNode"
	ctx, span := otel.Tracer("client").Start(ctx, method)
	defer span.End()
	traceID := span.SpanContext().TraceID().String()
	if c.Service == nil {
		return ErrClientNotReady
	}

	req := &milvuspb.TransferNodeRequest{
		SourceResourceGroup: sourceRg,
		TargetResourceGroup: targetRg,
		NumNode:             nodesNum,
	}

	resp, err := c.Service.TransferNode(ctx, req)
	if err != nil {
		log.Fatalf("transfer node failed, traceID:%s, error: %v", traceID, err)
		return err
	}
	return handleRespStatus(resp)
}

// TransferReplica transfer collection replicas between source,target resource group.
func (c *GrpcClient) TransferReplica(ctx context.Context, sourceRg, targetRg string, collectionName string, replicaNum int64) error {
	method := "TransferReplica"
	ctx, span := otel.Tracer("client").Start(ctx, method)
	defer span.End()
	traceID := span.SpanContext().TraceID().String()
	if c.Service == nil {
		return ErrClientNotReady
	}

	req := &milvuspb.TransferReplicaRequest{
		SourceResourceGroup: sourceRg,
		TargetResourceGroup: targetRg,
		CollectionName:      collectionName,
		NumReplica:          replicaNum,
	}

	resp, err := c.Service.TransferReplica(ctx, req)
	if err != nil {
		log.Fatalf("transfer replica failed, traceID:%s, error: %v", traceID, err)
		return err
	}
	return handleRespStatus(resp)
}
