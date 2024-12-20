package client

import (
	"context"
	"log"

	"go.opentelemetry.io/otel"

	"github.com/milvus-io/milvus-proto/go-api/v2/milvuspb"
	"github.com/milvus-io/milvus-sdk-go/v2/internal/utils/crypto"
)

// CreateCredential create new user and password
func (c *GrpcClient) CreateCredential(ctx context.Context, username string, password string) error {
	method := "CreateCredential"
	ctx, span := otel.Tracer("client").Start(ctx, method)
	defer span.End()
	traceID := span.SpanContext().TraceID().String()
	if c.Service == nil {
		return ErrClientNotReady
	}
	req := &milvuspb.CreateCredentialRequest{
		Username: username,
		Password: crypto.Base64Encode(password),
	}
	resp, err := c.Service.CreateCredential(ctx, req)
	if err != nil {
		log.Fatalf("create credential failed, traceID:%s err: %v", traceID, err)
		return err
	}
	err = handleRespStatus(resp)
	if err != nil {
		log.Fatalf("create credential failed, traceID:%s err: %v", traceID, err)
		return err
	}
	return nil
}

// UpdateCredential update password for a user
func (c *GrpcClient) UpdateCredential(ctx context.Context, username string, oldPassword string, newPassword string) error {
	method := "UpdateCredential"
	ctx, span := otel.Tracer("client").Start(ctx, method)
	defer span.End()
	traceID := span.SpanContext().TraceID().String()
	if c.Service == nil {
		return ErrClientNotReady
	}
	req := &milvuspb.UpdateCredentialRequest{
		Username:    username,
		OldPassword: crypto.Base64Encode(oldPassword),
		NewPassword: crypto.Base64Encode(newPassword),
	}
	resp, err := c.Service.UpdateCredential(ctx, req)
	if err != nil {
		log.Fatalf("update credential failed, traceID:%s err: %v", traceID, err)
		return err
	}
	err = handleRespStatus(resp)
	if err != nil {
		log.Fatalf("update credential failed, traceID:%s err: %v", traceID, err)
		return err
	}
	return nil
}

// DeleteCredential delete a user
func (c *GrpcClient) DeleteCredential(ctx context.Context, username string) error {
	method := "DeleteCredential"
	ctx, span := otel.Tracer("client").Start(ctx, method)
	defer span.End()
	traceID := span.SpanContext().TraceID().String()
	if c.Service == nil {
		return ErrClientNotReady
	}
	req := &milvuspb.DeleteCredentialRequest{
		Username: username,
	}
	resp, err := c.Service.DeleteCredential(ctx, req)
	if err != nil {
		log.Fatalf("delete credential failed, traceID:%s err: %v", traceID, err)
		return err
	}
	err = handleRespStatus(resp)
	if err != nil {
		log.Fatalf("delete credential failed, traceID:%s err: %v", traceID, err)
		return err
	}
	return nil
}

// ListCredUsers list all usernames
func (c *GrpcClient) ListCredUsers(ctx context.Context) ([]string, error) {
	method := "ListCredUsers"
	ctx, span := otel.Tracer("client").Start(ctx, method)
	defer span.End()
	traceID := span.SpanContext().TraceID().String()
	if c.Service == nil {
		return nil, ErrClientNotReady
	}
	req := &milvuspb.ListCredUsersRequest{}
	resp, err := c.Service.ListCredUsers(ctx, req)
	if err != nil {
		log.Fatalf("list credential users failed, traceID:%s err: %v", traceID, err)
		return nil, err
	}
	err = handleRespStatus(resp.Status)
	if err != nil {
		log.Fatalf("list credential users failed, traceID:%s err: %v", traceID, err)
		return nil, err
	}
	return resp.Usernames, nil
}
