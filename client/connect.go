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
	"os"
	"strconv"
	"time"

	"github.com/cockroachdb/errors"
	common "github.com/milvus-io/milvus-proto/go-api/v2/commonpb"
	server "github.com/milvus-io/milvus-proto/go-api/v2/milvuspb"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// GrpcClient, uses default grpc Service definition to connect with Milvus2.0
type GrpcClient struct {
	Conn    *grpc.ClientConn           // grpc connection instance
	Service server.MilvusServiceClient // Service client stub
	config  *Config                    // No thread safety
}

func (c *GrpcClient) dial(ctx context.Context, addr string, opts ...grpc.DialOption) error {
	if addr == "" {
		return fmt.Errorf("address is empty")
	}
	conn, err := grpc.DialContext(ctx, addr, opts...)
	if err != nil {
		return err
	}
	c.Conn = conn
	return nil
}

// connect connect to Service
func (c *GrpcClient) connect(ctx context.Context, addr string, opts ...grpc.DialOption) error {
	if err := c.dial(ctx, addr, opts...); err != nil {
		return err
	}
	c.Service = server.NewMilvusServiceClient(c.Conn)

	if !c.config.DisableConn {
		if err := c.connectInternal(ctx); err != nil {
			return err
		}
	}
	return nil
}

func (c *GrpcClient) connectInternal(ctx context.Context) error {
	hostName, err := os.Hostname()
	if err != nil {
		return err
	}

	req := &server.ConnectRequest{
		ClientInfo: &common.ClientInfo{
			SdkType:    "golang",
			SdkVersion: sdkVerion,
			LocalTime:  time.Now().String(),
			User:       c.config.Username,
			Host:       hostName,
		},
	}

	resp, err := c.Service.Connect(ctx, req)
	if err != nil {
		status, ok := status.FromError(err)
		if ok {
			if status.Code() == codes.Unimplemented {
				return errors.New("this version of sdk is incompatible with server," +
					" please downgrade your sdk or upgrade your server")
			}
		}
		return err
	}
	if err := handleRespStatus(resp.GetStatus()); err != nil {
		return err
	}

	c.config.Identifier = strconv.FormatInt(resp.GetIdentifier(), 10)
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
