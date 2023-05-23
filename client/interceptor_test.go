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
	testDBName = "TestDBName"
)

func TestAuthenticationInterceptor(t *testing.T) {
	ctx := context.Background()
	ctx = authenticationInterceptor(ctx, testUsername, testPassword)
	md, ok := metadata.FromOutgoingContext(ctx)
	value := crypto.Base64Encode(fmt.Sprintf("%s:%s", testUsername, testPassword))
	assert.True(t, ok)
	assert.Equal(t, 1, len(md["authorization"]))
	assert.Equal(t, value, md["authorization"][0])
}

func TestDBNameInterceptor(t *testing.T) {
	ctx := context.Background()
	ctx = databaseNameInterceptor(ctx, func() string { return testDBName })
	md, ok := metadata.FromOutgoingContext(ctx)
	assert.True(t, ok)
	assert.Equal(t, 1, len(md["dbname"]))
	assert.Equal(t, testDBName, md["dbname"][0])
}

func TestAPIInterceptor(t *testing.T) {
	ctx := context.Background()
	ctx = apiKeyInterceptor(ctx, "test-token")
	md, ok := metadata.FromOutgoingContext(ctx)
	value := crypto.Base64Encode(fmt.Sprintf("Bearer: %s", "test-token"))
	assert.True(t, ok)
	assert.Equal(t, 1, len(md[authorizationHeader]))
	assert.Equal(t, value, md[authorizationHeader][0])
}
