package client

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"google.golang.org/grpc/metadata"
)

func TestOutgoingCtx(t *testing.T) {
	t.Run("debug log level", func(t *testing.T) {
		ctx := WithDebugLogLevel(context.TODO())
		md, ok := metadata.FromOutgoingContext(ctx)
		assert.True(t, ok)
		assert.Equal(t, []string{"debug"}, md.Get(logLevelRPCMetaKey))
	})

	t.Run("info log level", func(t *testing.T) {
		ctx := WithInfoLogLevel(context.TODO())
		md, ok := metadata.FromOutgoingContext(ctx)
		assert.True(t, ok)
		assert.Equal(t, []string{"info"}, md.Get(logLevelRPCMetaKey))
	})

	t.Run("warn log level", func(t *testing.T) {
		ctx := WithWarnLogLevel(context.TODO())
		md, ok := metadata.FromOutgoingContext(ctx)
		assert.True(t, ok)
		assert.Equal(t, []string{"warn"}, md.Get(logLevelRPCMetaKey))
	})

	t.Run("error log level", func(t *testing.T) {
		ctx := WithErrorLogLevel(context.TODO())
		md, ok := metadata.FromOutgoingContext(ctx)
		assert.True(t, ok)
		assert.Equal(t, []string{"error"}, md.Get(logLevelRPCMetaKey))
	})

	t.Run("client request id", func(t *testing.T) {
		ctx := WithClientRequestID(context.TODO(), "test-trace")
		md, ok := metadata.FromOutgoingContext(ctx)
		assert.True(t, ok)
		assert.Equal(t, []string{"test-trace"}, md.Get(clientRequestIDKey))
	})
}
