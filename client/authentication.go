package client

import (
	"context"

	"github.com/milvus-io/milvus-proto/go-api/v2/milvuspb"
	"github.com/milvus-io/milvus-sdk-go/v2/internal/utils/crypto"
)

// CreateCredential create new user and password
func (c *GrpcClient) CreateCredential(ctx context.Context, username string, password string) error {
	service := c.Service(ctx)
	if service == nil {
		return ErrClientNotReady
	}
	defer service.Close()
	req := &milvuspb.CreateCredentialRequest{
		Username: username,
		Password: crypto.Base64Encode(password),
	}
	resp, err := service.CreateCredential(ctx, req)
	if err != nil {
		return err
	}
	err = handleRespStatus(resp)
	if err != nil {
		return err
	}
	return nil
}

// UpdateCredential update password for a user
func (c *GrpcClient) UpdateCredential(ctx context.Context, username string, oldPassword string, newPassword string) error {
	service := c.Service(ctx)
	if service == nil {
		return ErrClientNotReady
	}
	defer service.Close()
	req := &milvuspb.UpdateCredentialRequest{
		Username:    username,
		OldPassword: crypto.Base64Encode(oldPassword),
		NewPassword: crypto.Base64Encode(newPassword),
	}
	resp, err := service.UpdateCredential(ctx, req)
	if err != nil {
		return err
	}
	err = handleRespStatus(resp)
	if err != nil {
		return err
	}
	return nil
}

// DeleteCredential delete a user
func (c *GrpcClient) DeleteCredential(ctx context.Context, username string) error {
	service := c.Service(ctx)
	if service == nil {
		return ErrClientNotReady
	}
	defer service.Close()
	req := &milvuspb.DeleteCredentialRequest{
		Username: username,
	}
	resp, err := service.DeleteCredential(ctx, req)
	if err != nil {
		return err
	}
	err = handleRespStatus(resp)
	if err != nil {
		return err
	}
	return nil
}

// ListCredUsers list all usernames
func (c *GrpcClient) ListCredUsers(ctx context.Context) ([]string, error) {
	service := c.Service(ctx)
	if service == nil {
		return nil, ErrClientNotReady
	}
	defer service.Close()
	req := &milvuspb.ListCredUsersRequest{}
	resp, err := service.ListCredUsers(ctx, req)
	if err != nil {
		return nil, err
	}
	err = handleRespStatus(resp.Status)
	if err != nil {
		return nil, err
	}
	return resp.Usernames, nil
}
