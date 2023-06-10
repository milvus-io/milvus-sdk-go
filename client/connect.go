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
	"crypto/tls"
	"fmt"
	"math"
	"strconv"
	"sync/atomic"
	"time"

	grpc_middleware "github.com/grpc-ecosystem/go-grpc-middleware"
	grpc_retry "github.com/grpc-ecosystem/go-grpc-middleware/retry"
	common "github.com/milvus-io/milvus-proto/go-api/v2/commonpb"
	server "github.com/milvus-io/milvus-proto/go-api/v2/milvuspb"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/metadata"
)

// GrpcClient, uses default grpc Service definition to connect with Milvus2.0
type GrpcClient struct {
	Conn       *grpc.ClientConn           // grpc connection instance
	Service    server.MilvusServiceClient // Service client stub
	identifier atomic.Value
}

func (c *GrpcClient) getIdentifier() (int64, bool) {
	v, ok := c.identifier.Load().(int64)
	return v, ok
}

func (c GrpcClient) getIdentifierInterceptor() grpc.UnaryClientInterceptor {
	return func(ctx context.Context, method string, req, reply interface{}, cc *grpc.ClientConn, invoker grpc.UnaryInvoker, opts ...grpc.CallOption) error {
		identifier, ok := c.getIdentifier()
		if !ok {
			ctx = metadata.AppendToOutgoingContext(ctx, "identifier", strconv.FormatInt(identifier, 10))
		}
		return invoker(ctx, method, req, reply, cc, opts...)
	}
}

func (c *GrpcClient) getUnaryClientInteceptors() []grpc.UnaryClientInterceptor {
	return []grpc.UnaryClientInterceptor{
		c.getIdentifierInterceptor(),
	}
}

func (c *GrpcClient) dialWithOptions(ctx context.Context, addr string, dialOptions ...grpc.DialOption) (Client, error) {
	if err := c.connect(ctx, addr, dialOptions...); err != nil {
		return nil, err
	}
	return c, nil
}

func (c *GrpcClient) getDefaultDialOpts() []grpc.DialOption {
	clientInteceptors := c.getUnaryClientInteceptors()
	clientInteceptors = append(clientInteceptors, grpc_retry.UnaryClientInterceptor(
		grpc_retry.WithMax(6),
		grpc_retry.WithBackoff(func(attempt uint) time.Duration {
			return 60 * time.Millisecond * time.Duration(math.Pow(3, float64(attempt)))
		}),
		grpc_retry.WithCodes(codes.Unavailable, codes.ResourceExhausted)),
		RetryOnRateLimitInterceptor(10, func(ctx context.Context, attempt uint) time.Duration {
			return 10 * time.Millisecond * time.Duration(math.Pow(3, float64(attempt)))
		}))
	defaultOpts := append(DefaultGrpcOpts,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithUnaryInterceptor(
			grpc_middleware.ChainUnaryClient(
				clientInteceptors...,
			),
		),
	)

	return defaultOpts
}

func (c *GrpcClient) getDefaultAuthDialOpts(username, password string, enableTLS bool) []grpc.DialOption {
	var credential credentials.TransportCredentials
	if enableTLS {
		credential = credentials.NewTLS(&tls.Config{})
	} else {
		credential = insecure.NewCredentials()
	}

	clientInteceptors := c.getUnaryClientInteceptors()
	clientInteceptors = append(clientInteceptors,
		CreateAuthenticationUnaryInterceptor(username, password),
		grpc_retry.UnaryClientInterceptor(
			grpc_retry.WithMax(6),
			grpc_retry.WithBackoff(func(attempt uint) time.Duration {
				return 60 * time.Millisecond * time.Duration(math.Pow(3, float64(attempt)))
			}),
			grpc_retry.WithCodes(codes.Unavailable, codes.ResourceExhausted)),
	)
	defaultOpts := append(DefaultGrpcOpts,
		grpc.WithTransportCredentials(credential),
		grpc.WithChainUnaryInterceptor(clientInteceptors...),
		grpc.WithStreamInterceptor(CreateAuthenticationStreamInterceptor(username, password)),
	)
	return defaultOpts
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
	resp, err := c.Service.Connect(ctx, &server.ConnectRequest{
		Base: &common.MsgBase{},
		ClientInfo: &common.ClientInfo{
			SdkType:    "golang",
			SdkVersion: sdkVerion,
			//TODO get local ip and username if provided
		},
	})
	if err != nil {
		return err
	}
	if err := handleRespStatus(resp.GetStatus()); err != nil {
		return err
	}

	c.identifier.Store(resp.GetIdentifier())
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
