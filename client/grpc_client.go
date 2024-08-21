package client

import (
	"context"
	"fmt"
	"os"
	"strconv"
	"time"

	"golang.org/x/sync/singleflight"

	"github.com/milvus-io/milvus-proto/go-api/v2/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v2/milvuspb"
	"github.com/milvus-io/milvus-sdk-go/v2/common"
	"github.com/milvus-io/milvus-sdk-go/v2/merr"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// Check if GrpcClient implement Client.
var _ Client = &GrpcClient{}

// GrpcClient  uses default grpc Service definition to connect with Milvus2.0
type GrpcClient struct {
	Conn    *grpc.ClientConn             // grpc connection instance
	Service milvuspb.MilvusServiceClient // Service client stub

	cache *metaCache
	sf    singleflight.Group

	config *Config // No thread safety
}

// connect connect to Service
func (c *GrpcClient) connect(ctx context.Context, addr string, opts ...grpc.DialOption) error {
	if addr == "" {
		return fmt.Errorf("address is empty")
	}

	c.cache = newMetaCache()
	conn, err := grpc.DialContext(ctx, addr, opts...)
	if err != nil {
		return err
	}

	c.Conn = conn
	c.Service = milvuspb.NewMilvusServiceClient(c.Conn)

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

	req := &milvuspb.ConnectRequest{
		ClientInfo: &commonpb.ClientInfo{
			SdkType:    "Golang",
			SdkVersion: common.SDKVersion,
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
				return nil
			}
		}
		return err
	}

	if !merr.Ok(resp.GetStatus()) {
		return fmt.Errorf("connect fail, %s", resp.GetStatus().GetReason())
	}

	c.config.Identifier = strconv.FormatInt(resp.GetIdentifier(), 10)
	c.config.ServerVersion = resp.GetServerInfo().GetBuildTags()
	return nil
}

func (c *GrpcClient) getCollectionInfo(ctx context.Context, collectionName string) (*collInfo, error) {
	// try cache first
	info, ok := c.cache.getCollectionInfo(collectionName)
	if ok {
		return info, nil
	}

	// use singleflight to fetch collection info
	v, err, _ := c.sf.Do(collectionName, func() (interface{}, error) {
		coll, err := c.DescribeCollection(ctx, collectionName)
		if err != nil {
			return nil, err
		}
		return &collInfo{
			ID:               coll.ID,
			Name:             coll.Name,
			Schema:           coll.Schema,
			ConsistencyLevel: coll.ConsistencyLevel,
		}, nil
	})
	if err != nil {
		return nil, err
	}
	return v.(*collInfo), nil
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
