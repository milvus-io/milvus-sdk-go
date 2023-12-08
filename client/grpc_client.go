package client

import (
	"context"
	"fmt"
	"os"
	"strconv"
	"time"

	"github.com/milvus-io/milvus-proto/go-api/v2/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v2/milvuspb"
	"github.com/milvus-io/milvus-sdk-go/v2/common"
	grpcpool "github.com/processout/grpc-go-pool"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// Check if GrpcClient implement Client.
var _ Client = &GrpcClient{}

// GrpcClient  uses default grpc Service definition to connect with Milvus2.0
type GrpcClient struct {
	// Conn    *grpc.ClientConn             // grpc connection instance
	// Service milvuspb.MilvusServiceClient // Service client stub

	config *Config // No thread safety

	connPool *grpcpool.Pool
}

// connect connect to Service
func (c *GrpcClient) connect(ctx context.Context, addr string, opts ...grpc.DialOption) (*grpc.ClientConn, error) {
	if addr == "" {
		return nil, fmt.Errorf("address is empty")
	}
	conn, err := grpc.DialContext(ctx, addr, opts...)
	if err != nil {
		return nil, err
	}

	if !c.config.DisableConn {
		err = c.connectInternal(ctx, conn)
		if err != nil {
			return nil, err
		}
	}

	return conn, nil
}

func (c *GrpcClient) connectInternal(ctx context.Context, conn *grpc.ClientConn) error {
	hostName, err := os.Hostname()
	if err != nil {
		return err
	}

	req := &milvuspb.ConnectRequest{
		ClientInfo: &commonpb.ClientInfo{
			SdkType:    "Golang",
			SdkVersion: common.SDKVersion,
			LocalTime:  time.Now().String(),
			User:       c.config.Username,
			Host:       hostName,
		},
	}

	resp, err := c.getService(conn).Connect(ctx, req)
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
			return nil
		}
		return err
	}

	if resp.GetStatus().GetErrorCode() != commonpb.ErrorCode_Success {
		return fmt.Errorf("connect fail, %s", resp.Status.Reason)
	}

	c.config.Identifier = strconv.FormatInt(resp.GetIdentifier(), 10)
	c.config.ServerVersion = resp.GetServerInfo().GetBuildTags()
	return nil
}

func (c *GrpcClient) getService(conn *grpc.ClientConn) milvuspb.MilvusServiceClient {
	return milvuspb.NewMilvusServiceClient(conn)
}

func (c *GrpcClient) Service(ctx context.Context) *PoolingMilvusClient {
	if c.connPool == nil {
		return nil
	}
	conn, err := c.connPool.Get(ctx)
	if err != nil {
		return nil
	}
	return &PoolingMilvusClient{
		MilvusServiceClient: c.getService(conn.ClientConn),
		conn:                conn,
	}
}

type PoolingMilvusClient struct {
	milvuspb.MilvusServiceClient
	conn *grpcpool.ClientConn
}

func (c *PoolingMilvusClient) Close() {
	if c.conn != nil {
		c.conn.Close()
	}
}

// Close close the connection
func (c *GrpcClient) Close() error {
	if c.connPool != nil {
		c.connPool.Close()
	}
	return nil
}
