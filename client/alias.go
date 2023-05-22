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

	server "github.com/milvus-io/milvus-proto/go-api/milvuspb"
)

// CreateAlias creates an alias for collection
func (c *GrpcClient) CreateAlias(ctx context.Context, collName string, alias string) error {
	if c.Service == nil {
		return ErrClientNotReady
	}

	req := &server.CreateAliasRequest{
		DbName:         "", // reserved
		CollectionName: collName,
		Alias:          alias,
	}

	resp, err := c.Service.CreateAlias(ctx, req)
	if err != nil {
		return err
	}
	err = handleRespStatus(resp)
	if err != nil {
		return err
	}
	return nil
}

// DropAlias drops the specified Alias
func (c *GrpcClient) DropAlias(ctx context.Context, alias string) error {
	if c.Service == nil {
		return ErrClientNotReady
	}

	req := &server.DropAliasRequest{
		DbName: "", // reserved
		Alias:  alias,
	}

	resp, err := c.Service.DropAlias(ctx, req)
	if err != nil {
		return err
	}
	err = handleRespStatus(resp)
	if err != nil {
		return err
	}
	return nil
}

// AlterAlias changes collection alias to provided alias
func (c *GrpcClient) AlterAlias(ctx context.Context, collName string, alias string) error {
	if c.Service == nil {
		return ErrClientNotReady
	}

	req := &server.AlterAliasRequest{
		DbName:         "", // reserved
		CollectionName: collName,
		Alias:          alias,
	}

	resp, err := c.Service.AlterAlias(ctx, req)
	if err != nil {
		return err
	}
	err = handleRespStatus(resp)
	if err != nil {
		return err
	}
	return nil
}
