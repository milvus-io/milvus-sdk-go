package client

import (
	"context"
	"fmt"
	"os"
	"strconv"
	"time"

	common "github.com/milvus-io/milvus-proto/go-api/v2/commonpb"
	server "github.com/milvus-io/milvus-proto/go-api/v2/milvuspb"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// GrpcClient  uses default grpc Service definition to connect with Milvus2.0
type GrpcClient struct {
	Conn    *grpc.ClientConn           // grpc connection instance
	Service server.MilvusServiceClient // Service client stub

	config *Config // No thread safety
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

	if !c.config.DisableConn {
		err = c.connectInternal(ctx)
		if err != nil {
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
			SdkType:    "Golang",
			SdkVersion: "v2.2.4",
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
				// disable unsupported feature
				c.config.addFlags(
					disableDatabase |
						disableJSON |
						disableParitionKey |
						disableDynamicSchema)
			}
		} else {
			return err
		}
	}

	if resp.GetStatus().ErrorCode != common.ErrorCode_Success {
		return fmt.Errorf("connect fail, %s", resp.Status.Reason)
	}

	c.config.Identifier = strconv.FormatInt(resp.GetIdentifier(), 10)
	c.config.ServerVersion = resp.GetServerInfo().GetBuildTags()
	return nil
}
