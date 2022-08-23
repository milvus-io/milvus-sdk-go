package client

import (
	"context"

	"google.golang.org/grpc/metadata"
)

const (
	logLevelRPCMetaKey = "log_level"
	clientRequestIDKey = "client_request_id"

	debugLevel = "debug"
	infoLevel  = "info"
	warnLevel  = "warn"
	errorLevel = "error"
)

func WithDebugLogLevel(ctx context.Context) context.Context {
	return metadata.AppendToOutgoingContext(ctx, logLevelRPCMetaKey, debugLevel)
}

func WithInfoLogLevel(ctx context.Context) context.Context {
	return metadata.AppendToOutgoingContext(ctx, logLevelRPCMetaKey, infoLevel)
}

func WithWarnLogLevel(ctx context.Context) context.Context {
	return metadata.AppendToOutgoingContext(ctx, logLevelRPCMetaKey, warnLevel)
}

func WithErrorLogLevel(ctx context.Context) context.Context {
	return metadata.AppendToOutgoingContext(ctx, logLevelRPCMetaKey, errorLevel)
}

func WithClientRequestID(ctx context.Context, reqID string) context.Context {
	return metadata.AppendToOutgoingContext(ctx, clientRequestIDKey, reqID)
}
