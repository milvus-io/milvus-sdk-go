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

	"github.com/milvus-io/milvus-sdk-go/v2/entity"
)

// CreateDatabase creates a new database for remote Milvus cluster.
// TODO:New options can be added as expanding parameters.
func (c *GrpcClient) CreateDatabase(ctx context.Context, dbName string) error {
	if c.Service == nil {
		return ErrClientNotReady
	}
	req := &server.CreateDatabaseRequest{
		DbName: dbName,
	}
	resp, err := c.Service.CreateDatabase(ctx, req)
	if err != nil {
		return err
	}
	return handleRespStatus(resp)
}

// ListDatabases list all database in milvus cluster.
func (c *GrpcClient) ListDatabases(ctx context.Context) ([]entity.Database, error) {
	if c.Service == nil {
		return nil, ErrClientNotReady
	}

	req := &server.ListDatabasesRequest{}
	resp, err := c.Service.ListDatabases(ctx, req)
	if err != nil {
		return nil, err
	}
	if err = handleRespStatus(resp.GetStatus()); err != nil {
		return nil, err
	}
	databases := make([]entity.Database, len(resp.GetDbNames()))
	for i, dbName := range resp.GetDbNames() {
		databases[i] = entity.Database{
			Name: dbName,
		}
	}
	return databases, nil
}

// DropDatabase drop all database in milvus cluster.
func (c *GrpcClient) DropDatabase(ctx context.Context, dbName string) error {
	if c.Service == nil {
		return ErrClientNotReady
	}

	req := &server.DropDatabaseRequest{
		DbName: dbName,
	}
	resp, err := c.Service.DropDatabase(ctx, req)
	if err != nil {
		return err
	}
	return handleRespStatus(resp)
}
