package client

import (
	"context"
	"testing"

	"github.com/golang/protobuf/proto"
	"github.com/milvus-io/milvus-proto/go-api/v2/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v2/milvuspb"
	"github.com/stretchr/testify/assert"
)

func TestGrpcClientReplicateMessage(t *testing.T) {
	{
		ctx := context.Background()
		c := testClient(ctx, t)
		mockServer.SetInjection(MReplicateMessage, func(ctx context.Context, m proto.Message) (proto.Message, error) {
			return &milvuspb.ReplicateMessageResponse{
				Status:   &commonpb.Status{ErrorCode: commonpb.ErrorCode_Success},
				Position: "hello",
			}, nil
		})
		resp, err := c.ReplicateMessage(ctx, "ch1", 1000, 1001,
			nil, nil, nil, WithReplicateMessageMsgBase(&commonpb.MsgBase{}))
		assert.Nil(t, err)
		assert.Equal(t, "hello", resp.Position)
	}

	{
		ctx := context.Background()
		c := testClient(ctx, t)
		mockServer.SetInjection(MReplicateMessage, func(ctx context.Context, m proto.Message) (proto.Message, error) {
			return &milvuspb.ReplicateMessageResponse{
				Status: &commonpb.Status{ErrorCode: commonpb.ErrorCode_UnexpectedError},
			}, nil
		})
		_, err := c.ReplicateMessage(ctx, "ch1", 1000, 1001,
			nil, nil, nil, WithReplicateMessageMsgBase(&commonpb.MsgBase{}))
		assert.NotNil(t, err)
	}
}
