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
	"testing"

	"github.com/golang/protobuf/proto"
	"github.com/milvus-io/milvus-sdk-go/v2/internal/proto/common"
	"github.com/milvus-io/milvus-sdk-go/v2/internal/proto/server"
	"github.com/stretchr/testify/assert"
)

func TestGrpcCreateAlias(t *testing.T) {
	ctx := context.Background()
	c := testClient(ctx, t)
	defer c.Close()

	t.Run("normal create alias", func(t *testing.T) {

		mock.setInjection(mCreateAlias, func(_ context.Context, raw proto.Message) (proto.Message, error) {
			req, ok := raw.(*server.CreateAliasRequest)
			if !ok {
				t.FailNow()
			}
			assert.Equal(t, "testcoll", req.CollectionName)
			assert.Equal(t, "collAlias", req.Alias)

			return &common.Status{ErrorCode: common.ErrorCode_Success}, nil
		})
		defer mock.delInjection(mCreateAlias)
		err := c.CreateAlias(ctx, "testcoll", "collAlias")
		assert.NoError(t, err)
	})

	t.Run("alias duplicated", func(t *testing.T) {
		m := make(map[string]struct{})
		mock.setInjection(mCreateAlias, func(_ context.Context, raw proto.Message) (proto.Message, error) {
			req, ok := raw.(*server.CreateAliasRequest)
			if !ok {
				t.FailNow()
			}
			status := common.ErrorCode_Success
			_, has := m[req.GetAlias()]
			if has {
				status = common.ErrorCode_UnexpectedError
			}
			m[req.GetAlias()] = struct{}{}
			return &common.Status{ErrorCode: status}, nil
		})
		defer mock.delInjection(mCreateAlias)

		collName := "testColl"
		aliasName := "collAlias"
		err := c.CreateAlias(ctx, collName, aliasName)
		assert.NoError(t, err)
		err = c.CreateAlias(ctx, collName, aliasName)
		assert.Error(t, err)
	})
}

func TestGrpcDropAlias(t *testing.T) {
	ctx := context.Background()
	c := testClient(ctx, t)
	defer c.Close()

	t.Run("normal drop alias", func(t *testing.T) {
		mock.setInjection(mDropAlias, func(_ context.Context, raw proto.Message) (proto.Message, error) {
			req, ok := raw.(*server.DropAliasRequest)
			if !ok {
				t.FailNow()
			}
			assert.Equal(t, "collAlias", req.Alias)

			return &common.Status{ErrorCode: common.ErrorCode_Success}, nil
		})
		defer mock.delInjection(mDropAlias)
		err := c.DropAlias(ctx, "collAlias")
		assert.NoError(t, err)
	})

	t.Run("drop alias error", func(t *testing.T) {
		mock.setInjection(mDropAlias, func(_ context.Context, raw proto.Message) (proto.Message, error) {
			req, ok := raw.(*server.DropAliasRequest)
			if !ok {
				t.FailNow()
			}
			assert.Equal(t, "collAlias", req.Alias)

			return &common.Status{ErrorCode: common.ErrorCode_UnexpectedError}, nil
		})
		defer mock.delInjection(mDropAlias)
		err := c.DropAlias(ctx, "collAlias")
		assert.Error(t, err)
	})
}

func TestGrpcAlterAlias(t *testing.T) {
	ctx := context.Background()
	c := testClient(ctx, t)
	defer c.Close()

	collName := "collName"
	aliasName := "collAlias"

	t.Run("normal alter alias", func(t *testing.T) {
		mock.setInjection(mAlterAlias, func(_ context.Context, raw proto.Message) (proto.Message, error) {
			req, ok := raw.(*server.AlterAliasRequest)
			if !ok {
				t.FailNow()
			}
			assert.Equal(t, collName, req.CollectionName)
			assert.Equal(t, aliasName, req.Alias)

			return &common.Status{ErrorCode: common.ErrorCode_Success}, nil
		})
		defer mock.delInjection(mAlterAlias)
		err := c.AlterAlias(ctx, collName, aliasName)
		assert.NoError(t, err)
	})

	t.Run("alter alias error", func(t *testing.T) {
		mock.setInjection(mAlterAlias, func(_ context.Context, raw proto.Message) (proto.Message, error) {
			req, ok := raw.(*server.AlterAliasRequest)
			if !ok {
				t.FailNow()
			}
			assert.Equal(t, collName, req.CollectionName)
			assert.Equal(t, aliasName, req.Alias)

			return &common.Status{ErrorCode: common.ErrorCode_UnexpectedError}, nil
		})
		defer mock.delInjection(mAlterAlias)
		err := c.AlterAlias(ctx, collName, aliasName)
		assert.Error(t, err)
	})
}
