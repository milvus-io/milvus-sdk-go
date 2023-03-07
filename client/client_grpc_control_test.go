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
	common "github.com/milvus-io/milvus-proto/go-api/commonpb"
	server "github.com/milvus-io/milvus-proto/go-api/milvuspb"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
	"github.com/milvus-io/milvus-sdk-go/v2/internal/utils/tso"
	"github.com/stretchr/testify/assert"
)

func TestGrpcManualCompaction(t *testing.T) {
	ctx := context.Background()
	c := testClient(ctx, t)
	defer c.Close()

	mockServer.SetInjection(MHasCollection, hasCollectionDefault)
	defer mockServer.DelInjection(MHasCollection)

	compactionID := int64(1001)
	t.Run("normal manual compaction", func(t *testing.T) {
		now := time.Now()
		mockServer.SetInjection(MDescribeCollection, describeCollectionInjection(t, testCollectionID, testCollectionName, defaultSchema()))
		defer mockServer.DelInjection(MDescribeCollection)
		mockServer.SetInjection(MManualCompaction, func(ctx context.Context, raw proto.Message) (proto.Message, error) {
			req, ok := raw.(*server.ManualCompactionRequest)
			if !ok {
				t.FailNow()
			}

			assert.Equal(t, testCollectionID, req.GetCollectionID())
			ts, _ := tso.ParseTS(req.GetTimetravel())
			assert.True(t, ts.Sub(now) < time.Second)

			resp := &server.ManualCompactionResponse{
				CompactionID: compactionID,
			}
			s, err := SuccessStatus()
			resp.Status = s
			return resp, err
		})
		defer mockServer.DelInjection(MManualCompaction)

		id, err := c.ManualCompaction(ctx, testCollectionName, 0)
		assert.NoError(t, err)
		assert.EqualValues(t, compactionID, id)
	})

	t.Run("not ready client manual compaction", func(t *testing.T) {
		c := GrpcClient{}
		_, err := c.ManualCompaction(ctx, testCollectionName, 0)
		assert.Error(t, err)
		assert.Equal(t, ErrClientNotReady, err)
	})

	t.Run("describe collection fail", func(t *testing.T) {
		mockServer.SetInjection(MDescribeCollection, func(_ context.Context, _ proto.Message) (proto.Message, error) {
			return &server.DescribeCollectionResponse{}, errors.New("mocked error")
		})
		defer mockServer.DelInjection(MDescribeCollection)

		_, err := c.ManualCompaction(ctx, testCollectionName, 0)
		assert.Error(t, err)
	})

	t.Run("compaction Service error", func(t *testing.T) {
		mockServer.SetInjection(MDescribeCollection, describeCollectionInjection(t, testCollectionID, testCollectionName, defaultSchema()))
		defer mockServer.DelInjection(MDescribeCollection)
		mockServer.SetInjection(MManualCompaction, func(ctx context.Context, raw proto.Message) (proto.Message, error) {
			resp := &server.ManualCompactionResponse{
				CompactionID: 1001,
			}
			return resp, errors.New("mocked grpc error")
		})

		_, err := c.ManualCompaction(ctx, testCollectionName, 0)
		assert.Error(t, err)
		mockServer.SetInjection(MManualCompaction, func(ctx context.Context, raw proto.Message) (proto.Message, error) {
			resp := &server.ManualCompactionResponse{
				CompactionID: 1001,
			}
			resp.Status = &common.Status{
				ErrorCode: common.ErrorCode_UnexpectedError,
			}
			return resp, nil
		})

		defer mockServer.DelInjection(MManualCompaction)

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

		mockServer.SetInjection(MGetCompactionState, func(_ context.Context, raw proto.Message) (proto.Message, error) {
			req, ok := raw.(*server.GetCompactionStateRequest)
			if !ok {
				t.FailNow()
			}

			assert.Equal(t, compactionID, req.GetCompactionID())

			resp := &server.GetCompactionStateResponse{
				State: state,
			}
			s, err := SuccessStatus()
			resp.Status = s
			return resp, err
		})
		defer mockServer.DelInjection(MGetCompactionState)

		result, err := c.GetCompactionState(ctx, compactionID)
		assert.NoError(t, err)
		assert.Equal(t, entity.CompactionStateExecuting, result)

		state = common.CompactionState_Completed
		result, err = c.GetCompactionState(ctx, compactionID)
		assert.NoError(t, err)
		assert.Equal(t, entity.CompactionStateCompleted, result)
	})

	t.Run("get compaction Service fail", func(t *testing.T) {
		mockServer.SetInjection(MGetCompactionState, func(_ context.Context, raw proto.Message) (proto.Message, error) {
			resp := &server.GetCompactionStateResponse{}
			resp.Status = &common.Status{
				ErrorCode: common.ErrorCode_UnexpectedError,
			}
			return resp, nil
		})
		defer mockServer.DelInjection(MGetCompactionState)

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
			{Source: []int64{1, 2}, Target: 3, PlanType: entity.CompactionPlanMergeSegments},
			{Source: []int64{4, 5}, Target: 6, PlanType: entity.CompactionPlanMergeSegments},
		}

		mockServer.SetInjection(MGetCompactionStateWithPlans, func(_ context.Context, raw proto.Message) (proto.Message, error) {
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

			s, err := SuccessStatus()
			resp.Status = s
			return resp, err
		})
		defer mockServer.DelInjection(MGetCompactionStateWithPlans)

		result, rPlans, err := c.GetCompactionStateWithPlans(ctx, compactionID)
		assert.NoError(t, err)
		assert.Equal(t, entity.CompactionStateExecuting, result)
		assert.ElementsMatch(t, plans, rPlans)

		state = common.CompactionState_Completed
		result, rPlans, err = c.GetCompactionStateWithPlans(ctx, compactionID)
		assert.NoError(t, err)
		assert.Equal(t, entity.CompactionStateCompleted, result)
		assert.ElementsMatch(t, plans, rPlans)
	})

	t.Run("get compaction Service fail", func(t *testing.T) {
		mockServer.SetInjection(MGetCompactionStateWithPlans, func(_ context.Context, raw proto.Message) (proto.Message, error) {
			resp := &server.GetCompactionPlansResponse{}
			resp.Status = &common.Status{
				ErrorCode: common.ErrorCode_UnexpectedError,
			}
			return resp, nil
		})
		defer mockServer.DelInjection(MGetCompactionStateWithPlans)

		_, _, err := c.GetCompactionStateWithPlans(ctx, compactionID)
		assert.Error(t, err)
	})

}
