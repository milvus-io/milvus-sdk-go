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
	grpcpool "github.com/processout/grpc-go-pool"
	"google.golang.org/grpc"
)

// UsingDatabase for database operation after this function call.
// All request in any goroutine will be applied to new database on the same client. e.g.
// 1. goroutine A access DB1.
// 2. goroutine B call UsingDatabase(ctx, "DB2").
// 3. goroutine A access DB2 after 2.
func (c *GrpcClient) UsingDatabase(_ context.Context, dbName string) error {
	c.config.useDatabase(dbName)
	// Parse remote address.
	addr := c.config.getParsedAddress()

	// Parse grpc options
	options := c.config.getDialOption()

	factory := func() (*grpc.ClientConn, error) {
		ctx, cancel := context.WithTimeout(context.Background(), time.Second*3)
		defer cancel()
		return c.connect(ctx, addr, options...)
	}
	pool, err := grpcpool.New(factory, c.config.ConnPoolInit, c.config.ConnPoolMax, c.config.ConnPoolIdleTimeout)
	if err != nil {
		return err
	}
	c.connPool = pool
	return nil
}

// CreateDatabase creates a new database for remote Milvus cluster.
// TODO:New options can be added as expanding parameters.
func (c *GrpcClient) CreateDatabase(ctx context.Context, dbName string, opts ...CreateDatabaseOption) error {
	service := c.Service(ctx)
	if service == nil {
		return ErrClientNotReady
	}
	defer service.Close()
	if c.config.hasFlags(disableDatabase) {
		return ErrFeatureNotSupported
	}
	req := &milvuspb.CreateDatabaseRequest{
		DbName: dbName,
	}
	for _, opt := range opts {
		opt(req)
	}
	resp, err := service.CreateDatabase(ctx, req)
	if err != nil {
		return err
	}
	return handleRespStatus(resp)
}

// ListDatabases list all database in milvus cluster.
func (c *GrpcClient) ListDatabases(ctx context.Context) ([]entity.Database, error) {
	service := c.Service(ctx)
	if service == nil {
		return nil, ErrClientNotReady
	}
	defer service.Close()
	if c.config.hasFlags(disableDatabase) {
		return nil, ErrFeatureNotSupported
	}

	req := &milvuspb.ListDatabasesRequest{}
	resp, err := service.ListDatabases(ctx, req)
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
func (c *GrpcClient) DropDatabase(ctx context.Context, dbName string, opts ...DropDatabaseOption) error {
	service := c.Service(ctx)
	if service == nil {
		return ErrClientNotReady
	}
	defer service.Close()
	if c.config.hasFlags(disableDatabase) {
		return ErrFeatureNotSupported
	}

	req := &milvuspb.DropDatabaseRequest{
		DbName: dbName,
	}
	for _, opt := range opts {
		opt(req)
	}
	resp, err := service.DropDatabase(ctx, req)
	if err != nil {
		return err
	}
	return handleRespStatus(resp)
}
