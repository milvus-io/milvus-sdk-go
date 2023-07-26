package client

import (
	"context"
	"testing"

	"github.com/golang/protobuf/proto"
	common "github.com/milvus-io/milvus-proto/go-api/v2/commonpb"
	server "github.com/milvus-io/milvus-proto/go-api/v2/milvuspb"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
	"github.com/stretchr/testify/assert"
)

func TestCheckHealth(t *testing.T) {
	ctx := context.Background()
	c := testClient(ctx, t)
	defer c.Close()

	t.Run("test milvus healthy", func(t *testing.T) {
		mockServer.SetInjection(MCheckHealth, func(ctx context.Context, message proto.Message) (proto.Message, error) {
			resp := &server.CheckHealthResponse{
				Status:      &common.Status{ErrorCode: common.ErrorCode_Success},
				IsHealthy:   true,
				Reasons:     nil,
				QuotaStates: nil,
			}
			return resp, nil
		})
		defer mockServer.DelInjection(MCheckHealth)

		resp, err := c.CheckHealth(ctx)
		assert.Nil(t, err)
		assert.True(t, resp.IsHealthy)
		assert.Empty(t, resp.Reasons)
		assert.Empty(t, resp.QuotaStates)
	})

	t.Run("test milvus unhealthy", func(t *testing.T) {
		mockServer.SetInjection(MCheckHealth, func(ctx context.Context, message proto.Message) (proto.Message, error) {
			resp := &server.CheckHealthResponse{
				Status:      &common.Status{ErrorCode: common.ErrorCode_Success},
				IsHealthy:   false,
				Reasons:     []string{"some reason"},
				QuotaStates: []server.QuotaState{server.QuotaState_DenyToRead, server.QuotaState_DenyToWrite},
			}
			return resp, nil
		})
		defer mockServer.DelInjection(MCheckHealth)

		resp, err := c.CheckHealth(ctx)
		assert.Nil(t, err)
		assert.False(t, resp.IsHealthy)
		assert.ElementsMatch(t, resp.Reasons, []string{"some reason"})
		assert.ElementsMatch(t, resp.QuotaStates, []entity.QuotaState{entity.QuotaStateDenyToRead, entity.QuotaStateDenyToWrite})
	})

}
