package client

import (
	"context"
	"log"

	"go.opentelemetry.io/otel"

	"github.com/milvus-io/milvus-proto/go-api/v2/milvuspb"
	"github.com/milvus-io/milvus-proto/go-api/v2/msgpb"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
)

func (c *GrpcClient) ReplicateMessage(ctx context.Context,
	channelName string, beginTs, endTs uint64,
	msgsBytes [][]byte, startPositions, endPositions []*msgpb.MsgPosition,
	opts ...ReplicateMessageOption) (*entity.MessageInfo, error) {
	method := "ReplicateMessage"
	ctx, span := otel.Tracer("client").Start(ctx, method)
	defer span.End()
	traceID := span.SpanContext().TraceID().String()

	if c.Service == nil {
		return nil, ErrClientNotReady
	}
	req := &milvuspb.ReplicateMessageRequest{
		ChannelName:    channelName,
		BeginTs:        beginTs,
		EndTs:          endTs,
		Msgs:           msgsBytes,
		StartPositions: startPositions,
		EndPositions:   endPositions,
	}
	for _, opt := range opts {
		opt(req)
	}
	resp, err := c.Service.ReplicateMessage(ctx, req)
	if err != nil {
		log.Fatalf("replicate message failed, channelName:%s, traceID:%s err: %v", channelName, traceID, err)
		return nil, err
	}
	err = handleRespStatus(resp.GetStatus())
	if err != nil {
		log.Fatalf("replicate message failed, channelName:%s, traceID:%s err: %v", channelName, traceID, err)
		return nil, err
	}
	return &entity.MessageInfo{
		Position: resp.GetPosition(),
	}, nil
}
