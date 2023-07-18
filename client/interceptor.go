package client

import (
	"context"
	"fmt"

	"google.golang.org/grpc"
	"google.golang.org/grpc/metadata"

	"github.com/milvus-io/milvus-sdk-go/v2/internal/utils/crypto"
)

const (
	authorizationHeader = `authorization`

	identifierHeader = `identifier`

	databaseHeader = `dbname`
)

// authenticationInterceptor appends credential into context metadata
func authenticationInterceptor(ctx context.Context, username, password string) context.Context {
	if username != "" || password != "" {
		value := crypto.Base64Encode(fmt.Sprintf("%s:%s", username, password))
		return metadata.AppendToOutgoingContext(ctx, authorizationHeader, value)
	}
	return ctx
}

func apiKeyInterceptor(ctx context.Context, apiKey string) context.Context {
	if apiKey != "" {
		value := crypto.Base64Encode(apiKey)
		return metadata.AppendToOutgoingContext(ctx, authorizationHeader, value)
	}
	return ctx
}

func identifierInterceptor(ctx context.Context, identifierGetter func() string) context.Context {
	identifier := identifierGetter()
	if identifier != "" {
		ctx = metadata.AppendToOutgoingContext(ctx, identifierHeader, identifier)
	}
	return ctx
}

// databaseNameInterceptor appends the dbName into metadata.
func databaseNameInterceptor(ctx context.Context, dbNameGetter func() string) context.Context {
	dbname := dbNameGetter()
	if dbname != "" {
		ctx = metadata.AppendToOutgoingContext(ctx, databaseHeader, dbname)
	}
	return ctx
}

// createMetaDataUnaryInterceptor creates a unary interceptor for metadata information.
func createMetaDataUnaryInterceptor(cfg *Config) grpc.UnaryClientInterceptor {
	return func(ctx context.Context, method string, req, reply interface{}, cc *grpc.ClientConn, invoker grpc.UnaryInvoker, opts ...grpc.CallOption) error {
		ctx = authenticationInterceptor(ctx, cfg.Username, cfg.Password)
		ctx = apiKeyInterceptor(ctx, cfg.APIKey)
		ctx = identifierInterceptor(ctx, func() string {
			return cfg.Identifier
		})
		ctx = databaseNameInterceptor(ctx, func() string {
			return cfg.DBName
		})
		return invoker(ctx, method, req, reply, cc, opts...)
	}
}
