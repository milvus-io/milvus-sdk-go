package client

import (
	"context"
	"fmt"

	"github.com/milvus-io/milvus-sdk-go/v2/internal/utils/crypto"
	"google.golang.org/grpc"
	"google.golang.org/grpc/metadata"
)

const (
	authorizationHeader = `authorization`
)

// authenticationInterceptor appends credential into context metadata
func authenticationInterceptor(ctx context.Context, username, password string) context.Context {
	value := crypto.Base64Encode(fmt.Sprintf("%s:%s", username, password))
	return metadata.AppendToOutgoingContext(ctx, authorizationHeader, value)
}

func apiKeyInterceptor(ctx context.Context, apiKey string) context.Context {
	value := crypto.Base64Encode(fmt.Sprintf("Bearer: %s", apiKey))
	return metadata.AppendToOutgoingContext(ctx, authorizationHeader, value)
}

// CreateAuthenticationUnaryInterceptor creates a unary interceptor for basic authentication.
func createAuthenticationUnaryInterceptor(username, password string) grpc.UnaryClientInterceptor {
	return func(ctx context.Context, method string, req, reply interface{}, cc *grpc.ClientConn, invoker grpc.UnaryInvoker, opts ...grpc.CallOption) error {
		ctx = authenticationInterceptor(ctx, username, password)
		return invoker(ctx, method, req, reply, cc, opts...)
	}
}

// createAPIKeyUnaryInteceptor creates a unary inteceptor for api key authentication.
func createAPIKeyUnaryInteceptor(apiKey string) grpc.UnaryClientInterceptor {
	return func(ctx context.Context, method string, req, reply interface{}, cc *grpc.ClientConn, invoker grpc.UnaryInvoker, opts ...grpc.CallOption) error {
		ctx = apiKeyInterceptor(ctx, apiKey)
		return invoker(ctx, method, req, reply, cc, opts...)
	}
}

// createAuthenticationStreamInterceptor creates a stream interceptor for authentication
func createAuthenticationStreamInterceptor(username, password string) grpc.StreamClientInterceptor {
	return func(ctx context.Context, desc *grpc.StreamDesc, cc *grpc.ClientConn, method string, streamer grpc.Streamer, opts ...grpc.CallOption) (grpc.ClientStream, error) {
		ctx = authenticationInterceptor(ctx, username, password)
		return streamer(ctx, desc, cc, method, opts...)
	}
}

// databaseNameInterceptor appends the dbName into metadata.
func databaseNameInterceptor(ctx context.Context, dbNameGetter func() string) context.Context {
	dbname := dbNameGetter()
	if dbname != "" {
		ctx = metadata.AppendToOutgoingContext(ctx, "dbname", dbname)
	}
	return ctx
}

// createDatabaseNameInterceptor creates a unary interceptor for db name.
func createDatabaseNameUnaryInterceptor(dbNameGetter func() string) grpc.UnaryClientInterceptor {
	return func(ctx context.Context, method string, req, reply interface{}, cc *grpc.ClientConn, invoker grpc.UnaryInvoker, opts ...grpc.CallOption) error {
		ctx = databaseNameInterceptor(ctx, dbNameGetter)
		return invoker(ctx, method, req, reply, cc, opts...)
	}
}

// createDatabaseNameStreamInterceptor creates a unary interceptor for db name.
func createDatabaseNameStreamInterceptor(dbNameGetter func() string) grpc.StreamClientInterceptor {
	return func(ctx context.Context, desc *grpc.StreamDesc, cc *grpc.ClientConn, method string, streamer grpc.Streamer, opts ...grpc.CallOption) (grpc.ClientStream, error) {
		ctx = databaseNameInterceptor(ctx, dbNameGetter)
		return streamer(ctx, desc, cc, method, opts...)
	}
}
