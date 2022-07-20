package client

import (
	"context"
	"fmt"
	"testing"

	"github.com/milvus-io/milvus-sdk-go/v2/internal/utils/crypto"
	"github.com/stretchr/testify/assert"
	"google.golang.org/grpc/metadata"
)

const (
	TestUsername = "TestUsername"
	TestPassword = "TestPassword"
)

func TestAuthenticationInterceptor(t *testing.T) {
	ctx := context.Background()
	ctx = AuthenticationInterceptor(ctx, TestUsername, TestPassword)
	md, ok := metadata.FromOutgoingContext(ctx)
	value := crypto.Base64Encode(fmt.Sprintf("%s:%s", TestUsername, TestPassword))
	assert.True(t, ok)
	assert.Equal(t, 1, len(md["authorization"]))
	assert.Equal(t, value, md["authorization"][0])
}
