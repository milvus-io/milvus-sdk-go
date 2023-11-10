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
	"fmt"
	"time"

	"github.com/cockroachdb/errors"

	"github.com/milvus-io/milvus-proto/go-api/v2/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v2/milvuspb"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
)

// CreatePartition create partition for collection
func (c *GrpcClient) CreatePartition(ctx context.Context, collName string, partitionName string, opts ...CreatePartitionOption) error {
	if c.Service == nil {
		return ErrClientNotReady
	}
	if err := c.checkCollectionExists(ctx, collName); err != nil {
		return err
	}
	has, err := c.HasPartition(ctx, collName, partitionName)
	if err != nil {
		return err
	}
	if has {
		return fmt.Errorf("partition %s of collection %s already exists", partitionName, collName)
	}

	req := &milvuspb.CreatePartitionRequest{
		DbName:         "", //reserved
		CollectionName: collName,
		PartitionName:  partitionName,
	}
	for _, opt := range opts {
		opt(req)
	}
	resp, err := c.Service.CreatePartition(ctx, req)
	if err != nil {
		return err
	}
	return handleRespStatus(resp)
}

func (c *GrpcClient) checkPartitionExists(ctx context.Context, collName string, partitionName string) error {
	has, err := c.HasPartition(ctx, collName, partitionName)
	if err != nil {
		return err
	}
	if !has {
		return partNotExistsErr(collName, partitionName)
	}
	return nil
}

// DropPartition drop partition from collection
func (c *GrpcClient) DropPartition(ctx context.Context, collName string, partitionName string, opts ...DropPartitionOption) error {
	if c.Service == nil {
		return ErrClientNotReady
	}
	if err := c.checkCollectionExists(ctx, collName); err != nil {
		return err
	}
	if err := c.checkPartitionExists(ctx, collName, partitionName); err != nil {
		return err
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
		return err
	}
	return handleRespStatus(resp)
}

// HasPartition check whether specified partition exists
func (c *GrpcClient) HasPartition(ctx context.Context, collName string, partitionName string) (bool, error) {
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
		return false, err
	}
	if resp.GetStatus().GetErrorCode() != commonpb.ErrorCode_Success {
		return false, errors.New("request failed")
	}
	return resp.GetValue(), nil
}

// ShowPartitions list all partitions from collection
func (c *GrpcClient) ShowPartitions(ctx context.Context, collName string) ([]*entity.Partition, error) {
	if c.Service == nil {
		return []*entity.Partition{}, ErrClientNotReady
	}
	req := &milvuspb.ShowPartitionsRequest{
		DbName:         "", // reserved
		CollectionName: collName,
	}
	resp, err := c.Service.ShowPartitions(ctx, req)
	if err != nil {
		return []*entity.Partition{}, err
	}
	if err := handleRespStatus(resp.GetStatus()); err != nil {
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
	if c.Service == nil {
		return ErrClientNotReady
	}
	if err := c.checkCollectionExists(ctx, collName); err != nil {
		return err
	}
	for _, partitionName := range partitionNames {
		if err := c.checkPartitionExists(ctx, collName, partitionName); err != nil {
			return err
		}
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
		return err
	}
	if err := handleRespStatus(resp); err != nil {
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
	if c.Service == nil {
		return ErrClientNotReady
	}
	if err := c.checkCollectionExists(ctx, collName); err != nil {
		return err
	}
	for _, partitionName := range partitionNames {
		if err := c.checkPartitionExists(ctx, collName, partitionName); err != nil {
			return err
		}
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
		return err
	}

	return handleRespStatus(resp)
}
