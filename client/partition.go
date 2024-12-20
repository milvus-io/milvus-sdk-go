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
	"time"

	"github.com/cockroachdb/errors"
	"go.opentelemetry.io/otel"

	"github.com/milvus-io/milvus-proto/go-api/v2/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v2/milvuspb"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
)

// CreatePartition create partition for collection
func (c *GrpcClient) CreatePartition(ctx context.Context, collName string, partitionName string, opts ...CreatePartitionOption) error {
	method := "CreatePartition"
	ctx, span := otel.Tracer("client").Start(ctx, method)
	defer span.End()
	traceID := span.SpanContext().TraceID().String()

	if c.Service == nil {
		return ErrClientNotReady
	}

	req := &milvuspb.CreatePartitionRequest{
		DbName:         "", // reserved
		CollectionName: collName,
		PartitionName:  partitionName,
	}
	for _, opt := range opts {
		opt(req)
	}
	resp, err := c.Service.CreatePartition(ctx, req)
	if err != nil {
		log.Fatalf("create partition failed, collName:%s, partitionName:%s, traceID:%s err: %v", collName, partitionName, traceID, err)
		return err
	}
	return handleRespStatus(resp)
}

// DropPartition drop partition from collection
func (c *GrpcClient) DropPartition(ctx context.Context, collName string, partitionName string, opts ...DropPartitionOption) error {
	method := "DropPartition"
	ctx, span := otel.Tracer("client").Start(ctx, method)
	defer span.End()
	traceID := span.SpanContext().TraceID().String()
	if c.Service == nil {
		return ErrClientNotReady
	}

	req := &milvuspb.DropPartitionRequest{
		DbName:         "",
		CollectionName: collName,
		PartitionName:  partitionName,
	}
	for _, opt := range opts {
		opt(req)
	}
	resp, err := c.Service.DropPartition(ctx, req)
	if err != nil {
		log.Fatalf("drop partition failed, collName:%s, partitionName:%s, traceID:%s err: %v", collName, partitionName, traceID, err)
		return err
	}
	return handleRespStatus(resp)
}

// HasPartition check whether specified partition exists
func (c *GrpcClient) HasPartition(ctx context.Context, collName string, partitionName string) (bool, error) {
	method := "HasPartition"
	ctx, span := otel.Tracer("client").Start(ctx, method)
	defer span.End()
	traceID := span.SpanContext().TraceID().String()
	if c.Service == nil {
		return false, ErrClientNotReady
	}
	req := &milvuspb.HasPartitionRequest{
		DbName:         "", // reserved
		CollectionName: collName,
		PartitionName:  partitionName,
	}
	resp, err := c.Service.HasPartition(ctx, req)
	if err != nil {
		log.Fatalf("has partition failed, collName:%s, partitionName:%s, traceID:%s err: %v", collName, partitionName, traceID, err)
		return false, err
	}
	if resp.GetStatus().GetErrorCode() != commonpb.ErrorCode_Success {
		return false, errors.New("request failed")
	}
	return resp.GetValue(), nil
}

// ShowPartitions list all partitions from collection
func (c *GrpcClient) ShowPartitions(ctx context.Context, collName string) ([]*entity.Partition, error) {
	method := "ShowPartitions"
	ctx, span := otel.Tracer("client").Start(ctx, method)
	defer span.End()
	traceID := span.SpanContext().TraceID().String()
	if c.Service == nil {
		return []*entity.Partition{}, ErrClientNotReady
	}
	req := &milvuspb.ShowPartitionsRequest{
		DbName:         "", // reserved
		CollectionName: collName,
	}
	resp, err := c.Service.ShowPartitions(ctx, req)
	if err != nil {
		log.Fatalf("show partitions failed, collName:%s, traceID:%s err: %v", collName, traceID, err)
		return []*entity.Partition{}, err
	}
	if err := handleRespStatus(resp.GetStatus()); err != nil {
		log.Fatalf("show partitions failed, collName:%s, traceID:%s err: %v", collName, traceID, err)
		return []*entity.Partition{}, err
	}
	partitions := make([]*entity.Partition, 0, len(resp.GetPartitionIDs()))
	if len(resp.GetPartitionNames()) == 0 {
		return []*entity.Partition{}, errors.New("length of PartitionNames")
	}
	for idx, partitionID := range resp.GetPartitionIDs() {
		partition := &entity.Partition{ID: partitionID, Name: resp.GetPartitionNames()[idx]}
		if len(resp.GetInMemoryPercentages()) > idx {
			partition.Loaded = resp.GetInMemoryPercentages()[idx] == 100
		}

		partitions = append(partitions, partition)
	}
	return partitions, nil
}

// LoadPartitions load collection paritions into memory
func (c *GrpcClient) LoadPartitions(ctx context.Context, collName string, partitionNames []string, async bool, opts ...LoadPartitionsOption) error {
	method := "LoadPartitions"
	ctx, span := otel.Tracer("client").Start(ctx, method)
	defer span.End()
	traceID := span.SpanContext().TraceID().String()
	if c.Service == nil {
		return ErrClientNotReady
	}

	req := &milvuspb.LoadPartitionsRequest{
		DbName:         "", // reserved
		CollectionName: collName,
		PartitionNames: partitionNames,
	}
	for _, opt := range opts {
		opt(req)
	}
	resp, err := c.Service.LoadPartitions(ctx, req)
	if err != nil {
		log.Fatalf("load partitions failed, collName:%s, traceID:%s err: %v", collName, traceID, err)
		return err
	}
	if err := handleRespStatus(resp); err != nil {
		log.Fatalf("load partitions failed, collName:%s, traceID:%s err: %v", collName, traceID, err)
		return err
	}

	if !async {
		ticker := time.NewTicker(200 * time.Millisecond)
		defer ticker.Stop()

		for {
			select {
			case <-ctx.Done():
				return ctx.Err()
			case <-ticker.C:
				progress, err := c.getLoadingProgress(ctx, collName, partitionNames...)
				if err != nil {
					log.Fatalf("get loading progress failed, traceID:%s err: %v", traceID, err)
					return err
				}
				if progress == 100 {
					return nil
				}
			}
		}
	}

	return nil
}

// ReleasePartitions release partitions
func (c *GrpcClient) ReleasePartitions(ctx context.Context, collName string, partitionNames []string, opts ...ReleasePartitionsOption) error {
	method := "ReleasePartitions"
	ctx, span := otel.Tracer("client").Start(ctx, method)
	defer span.End()
	traceID := span.SpanContext().TraceID().String()
	if c.Service == nil {
		return ErrClientNotReady
	}
	req := &milvuspb.ReleasePartitionsRequest{
		DbName:         "", // reserved
		CollectionName: collName,
		PartitionNames: partitionNames,
	}
	for _, opt := range opts {
		opt(req)
	}
	resp, err := c.Service.ReleasePartitions(ctx, req)
	if err != nil {
		log.Fatalf("release partitions failed, collName:%s, partitionNames:%v, traceID:%s err: %v", collName, partitionNames, traceID, err)
		return err
	}

	return handleRespStatus(resp)
}
