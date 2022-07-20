package client

import (
	"context"
	"fmt"

	"github.com/milvus-io/milvus-sdk-go/v2/internal/utils/crypto"
	"google.golang.org/grpc"
	"google.golang.org/grpc/metadata"
)

// AuthenticationInterceptor appends credential into context metadata
func AuthenticationInterceptor(ctx context.Context, username, password string) context.Context {
	value := crypto.Base64Encode(fmt.Sprintf("%s:%s", username, password))
	return metadata.AppendToOutgoingContext(ctx, "authorization", value)
}

// CreateAuthenticationUnaryInterceptor creates a unary interceptor for authentication
func CreateAuthenticationUnaryInterceptor(username, password string) grpc.UnaryClientInterceptor {
	return func(ctx context.Context, method string, req, reply interface{}, cc *grpc.ClientConn, invoker grpc.UnaryInvoker, opts ...grpc.CallOption) error {
		ctx = AuthenticationInterceptor(ctx, username, password)
		return invoker(ctx, method, req, reply, cc, opts...)
	}
}

// CreateAuthenticationStreamInterceptor creates a stream interceptor for authentication
func CreateAuthenticationStreamInterceptor(username, password string) grpc.StreamClientInterceptor {
	return func(ctx context.Context, desc *grpc.StreamDesc, cc *grpc.ClientConn, method string, streamer grpc.Streamer, opts ...grpc.CallOption) (grpc.ClientStream, error) {
		ctx = AuthenticationInterceptor(ctx, username, password)
		return streamer(ctx, desc, cc, method, opts...)
	}
}
