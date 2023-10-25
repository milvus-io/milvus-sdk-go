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

// GetVersion returns milvus server version information.
func (c *GrpcClient) GetVersion(ctx context.Context) (string, error) {
	if c.Service == nil {
		return "", ErrClientNotReady
	}
	resp, err := c.Service.GetVersion(ctx, &milvuspb.GetVersionRequest{})
	if err != nil {
		return "", err
	}
	return resp.GetVersion(), nil
}

// CheckHealth returns milvus state
func (c *GrpcClient) CheckHealth(ctx context.Context) (*entity.MilvusState, error) {
	if c.Service == nil {
		return nil, ErrClientNotReady
	}
	resp, err := c.Service.CheckHealth(ctx, &milvuspb.CheckHealthRequest{})
	if err != nil {
		return nil, err
	}

	states := make([]entity.QuotaState, 0, len(resp.GetQuotaStates()))
	for _, state := range resp.GetQuotaStates() {
		states = append(states, entity.QuotaState(state))
	}

	return &entity.MilvusState{
		IsHealthy:   resp.GetIsHealthy(),
		Reasons:     resp.GetReasons(),
		QuotaStates: states,
	}, nil
}
