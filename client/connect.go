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

	server "github.com/milvus-io/milvus-proto/go-api/milvuspb"
	"google.golang.org/grpc"
)

// GrpcClient, uses default grpc Service definition to connect with Milvus2.0
type GrpcClient struct {
	Conn    *grpc.ClientConn           // grpc connection instance
	Service server.MilvusServiceClient // Service client stub
}

// connect connect to Service
func (c *GrpcClient) connect(ctx context.Context, addr string, opts ...grpc.DialOption) error {
	if addr == "" {
		return fmt.Errorf("address is empty")
	}
	conn, err := grpc.DialContext(ctx, addr, opts...)
	if err != nil {
		return err
	}

	c.Conn = conn
	c.Service = server.NewMilvusServiceClient(c.Conn)
	return nil
}

// Close close the connection
func (c *GrpcClient) Close() error {
	if c.Conn != nil {
		err := c.Conn.Close()
		c.Conn = nil
		return err
	}
	return nil
}
