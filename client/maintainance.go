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
	"time"

	"github.com/milvus-io/milvus-proto/go-api/v2/milvuspb"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
)

// ManualCompaction triggers a compaction on provided collection
func (c *GrpcClient) ManualCompaction(ctx context.Context, collName string, _ time.Duration) (int64, error) {
	service := c.Service(ctx)
	if service == nil {
		return 0, ErrClientNotReady
	}
	defer service.Close()

	if err := c.checkCollectionExists(ctx, service, collName); err != nil {
		return 0, err
	}
	coll, err := c.requestDescribeCollection(ctx, service, collName)
	if err != nil {
		return 0, err
	}

	req := &milvuspb.ManualCompactionRequest{
		CollectionID: coll.ID,
	}

	resp, err := service.ManualCompaction(ctx, req)
	if err != nil {
		return 0, err
	}

	err = handleRespStatus(resp.GetStatus())
	if err != nil {
		return 0, err
	}

	return resp.GetCompactionID(), nil
}

// GetCompactionState get compaction state of provided compaction id
func (c *GrpcClient) GetCompactionState(ctx context.Context, id int64) (entity.CompactionState, error) {
	service := c.Service(ctx)
	if service == nil {
		return entity.CompcationStateUndefined, ErrClientNotReady
	}
	defer service.Close()

	req := &milvuspb.GetCompactionStateRequest{CompactionID: id}
	resp, err := service.GetCompactionState(ctx, req)
	if err != nil {
		return entity.CompcationStateUndefined, err
	}

	err = handleRespStatus(resp.GetStatus())
	if err != nil {
		return entity.CompcationStateUndefined, err
	}

	// direct mapping values of CompactionState
	return entity.CompactionState(resp.GetState()), nil
}

// GetCompactionStateWithPlans get compaction state with plans of provided compaction id
func (c *GrpcClient) GetCompactionStateWithPlans(ctx context.Context, id int64) (entity.CompactionState, []entity.CompactionPlan, error) {
	service := c.Service(ctx)
	if service == nil {
		return entity.CompcationStateUndefined, nil, ErrClientNotReady
	}
	defer service.Close()

	req := &milvuspb.GetCompactionPlansRequest{
		CompactionID: id,
	}
	resp, err := service.GetCompactionStateWithPlans(ctx, req)
	if err != nil {
		return entity.CompcationStateUndefined, nil, err
	}

	err = handleRespStatus(resp.GetStatus())
	if err != nil {
		return entity.CompcationStateUndefined, nil, err
	}

	plans := make([]entity.CompactionPlan, 0, len(resp.GetMergeInfos()))
	for _, mergeInfo := range resp.GetMergeInfos() {
		plans = append(plans, entity.CompactionPlan{
			Source:   mergeInfo.GetSources(),
			Target:   mergeInfo.GetTarget(),
			PlanType: entity.CompactionPlanMergeSegments,
		})
	}

	return entity.CompactionState(resp.GetState()), plans, nil
}
