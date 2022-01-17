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
	"testing"
	"time"

	"github.com/golang/protobuf/proto"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
	"github.com/milvus-io/milvus-sdk-go/v2/internal/proto/common"
	"github.com/milvus-io/milvus-sdk-go/v2/internal/proto/server"
	"github.com/milvus-io/milvus-sdk-go/v2/internal/utils/tso"
	"github.com/stretchr/testify/assert"
)

func TestGrpcManualCompaction(t *testing.T) {
	ctx := context.Background()
	c := testClient(ctx, t)
	defer c.Close()

	collID := int64(1)
	compactionID := int64(1001)
	t.Run("normal manual compaction", func(t *testing.T) {
		now := time.Now()
		mock.setInjection(mDescribeCollection, describeCollectionInjection(t, collID, testCollectionName, defaultSchema()))
		defer mock.delInjection(mDescribeCollection)
		mock.setInjection(mManualCompaction, func(ctx context.Context, raw proto.Message) (proto.Message, error) {
			req, ok := raw.(*server.ManualCompactionRequest)
			if !ok {
				t.FailNow()
			}

			assert.Equal(t, collID, req.GetCollectionID())
			ts, _ := tso.ParseTS(req.GetTimetravel())
			assert.True(t, ts.Sub(now) < time.Second)

			resp := &server.ManualCompactionResponse{
				CompactionID: compactionID,
			}
			s, err := successStatus()
			resp.Status = s
			return resp, err
		})
		defer mock.delInjection(mManualCompaction)

		id, err := c.ManualCompaction(ctx, testCollectionName, 0)
		assert.NoError(t, err)
		assert.EqualValues(t, compactionID, id)
	})

	t.Run("not ready client manual compaction", func(t *testing.T) {
		c := grpcClient{}
		_, err := c.ManualCompaction(ctx, testCollectionName, 0)
		assert.Error(t, err)
		assert.Equal(t, ErrClientNotReady, err)
	})

	t.Run("describe collection fail", func(t *testing.T) {
		mock.setInjection(mDescribeCollection, func(_ context.Context, _ proto.Message) (proto.Message, error) {
			return &server.DescribeCollectionResponse{}, errors.New("mocked error")
		})
		defer mock.delInjection(mDescribeCollection)

		_, err := c.ManualCompaction(ctx, testCollectionName, 0)
		assert.Error(t, err)
	})

	t.Run("compaction service error", func(t *testing.T) {
		mock.setInjection(mDescribeCollection, describeCollectionInjection(t, collID, testCollectionName, defaultSchema()))
		defer mock.delInjection(mDescribeCollection)
		mock.setInjection(mManualCompaction, func(ctx context.Context, raw proto.Message) (proto.Message, error) {
			resp := &server.ManualCompactionResponse{
				CompactionID: 1001,
			}
			return resp, errors.New("mocked grpc error")
		})

		_, err := c.ManualCompaction(ctx, testCollectionName, 0)
		assert.Error(t, err)
		mock.setInjection(mManualCompaction, func(ctx context.Context, raw proto.Message) (proto.Message, error) {
			resp := &server.ManualCompactionResponse{
				CompactionID: 1001,
			}
			resp.Status = &common.Status{
				ErrorCode: common.ErrorCode_UnexpectedError,
			}
			return resp, nil
		})

		defer mock.delInjection(mManualCompaction)

		_, err = c.ManualCompaction(ctx, testCollectionName, 0)
		assert.Error(t, err)
	})
}

func TestGrpcGetCompactionState(t *testing.T) {
	ctx := context.Background()
	c := testClient(ctx, t)
	defer c.Close()

	compactionID := int64(1001)

	t.Run("normal get compaction state", func(t *testing.T) {
		state := common.CompactionState_Executing

		mock.setInjection(mGetCompactionState, func(_ context.Context, raw proto.Message) (proto.Message, error) {
			req, ok := raw.(*server.GetCompactionStateRequest)
			if !ok {
				t.FailNow()
			}

			assert.Equal(t, compactionID, req.GetCompactionID())

			resp := &server.GetCompactionStateResponse{
				State: state,
			}
			s, err := successStatus()
			resp.Status = s
			return resp, err
		})
		defer mock.delInjection(mGetCompactionState)

		result, err := c.GetCompactionState(ctx, compactionID)
		assert.NoError(t, err)
		assert.Equal(t, entity.COMPACTION_STATE_EXECUTING, result)

		state = common.CompactionState_Completed
		result, err = c.GetCompactionState(ctx, compactionID)
		assert.NoError(t, err)
		assert.Equal(t, entity.COMPACTION_STATE_COMPLETED, result)
	})

	t.Run("get compaction service fail", func(t *testing.T) {
		mock.setInjection(mGetCompactionState, func(_ context.Context, raw proto.Message) (proto.Message, error) {
			resp := &server.GetCompactionStateResponse{}
			resp.Status = &common.Status{
				ErrorCode: common.ErrorCode_UnexpectedError,
			}
			return resp, nil
		})
		defer mock.delInjection(mGetCompactionState)

		_, err := c.GetCompactionState(ctx, compactionID)
		assert.Error(t, err)
	})
}

func TestGrpcGetCompactionStateWithPlans(t *testing.T) {
	ctx := context.Background()
	c := testClient(ctx, t)
	defer c.Close()

	compactionID := int64(1001)

	t.Run("normal get compaction state with plans", func(t *testing.T) {
		state := common.CompactionState_Executing
		plans := []entity.CompactionPlan{
			{Source: []int64{1, 2}, Target: 3, PlanType: entity.COMPACTION_PLAN_MERGE_SEGMENTS},
			{Source: []int64{4, 5}, Target: 6, PlanType: entity.COMPACTION_PLAN_MERGE_SEGMENTS},
		}

		mock.setInjection(mGetCompactionStateWithPlans, func(_ context.Context, raw proto.Message) (proto.Message, error) {
			req, ok := raw.(*server.GetCompactionPlansRequest)
			if !ok {
				t.FailNow()
			}

			assert.Equal(t, compactionID, req.GetCompactionID())

			resp := &server.GetCompactionPlansResponse{
				State:      state,
				MergeInfos: make([]*server.CompactionMergeInfo, 0, len(plans)),
			}
			for _, plan := range plans {
				resp.MergeInfos = append(resp.MergeInfos, &server.CompactionMergeInfo{
					Sources: plan.Source,
					Target:  plan.Target,
				})
			}

			s, err := successStatus()
			resp.Status = s
			return resp, err
		})
		defer mock.delInjection(mGetCompactionStateWithPlans)

		result, rPlans, err := c.GetCompactionStateWithPlans(ctx, compactionID)
		assert.NoError(t, err)
		assert.Equal(t, entity.COMPACTION_STATE_EXECUTING, result)
		assert.ElementsMatch(t, plans, rPlans)

		state = common.CompactionState_Completed
		result, rPlans, err = c.GetCompactionStateWithPlans(ctx, compactionID)
		assert.NoError(t, err)
		assert.Equal(t, entity.COMPACTION_STATE_COMPLETED, result)
		assert.ElementsMatch(t, plans, rPlans)
	})

	t.Run("get compaction service fail", func(t *testing.T) {
		mock.setInjection(mGetCompactionStateWithPlans, func(_ context.Context, raw proto.Message) (proto.Message, error) {
			resp := &server.GetCompactionPlansResponse{}
			resp.Status = &common.Status{
				ErrorCode: common.ErrorCode_UnexpectedError,
			}
			return resp, nil
		})
		defer mock.delInjection(mGetCompactionStateWithPlans)

		_, _, err := c.GetCompactionStateWithPlans(ctx, compactionID)
		assert.Error(t, err)
	})

}
