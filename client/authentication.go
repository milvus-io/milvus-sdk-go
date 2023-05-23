package client

import (
	"context"

	server "github.com/milvus-io/milvus-proto/go-api/milvuspb"
	"github.com/milvus-io/milvus-sdk-go/v2/internal/utils/crypto"
)

// CreateCredential create new user and password
func (c *GrpcClient) CreateCredential(ctx context.Context, username string, password string) error {
	if c.Service == nil {
		return ErrClientNotReady
	}
	req := &server.CreateCredentialRequest{
		Username: username,
		Password: crypto.Base64Encode(password),
	}
	resp, err := c.Service.CreateCredential(ctx, req)
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
	if c.Service == nil {
		return ErrClientNotReady
	}
	req := &server.UpdateCredentialRequest{
		Username:    username,
		OldPassword: crypto.Base64Encode(oldPassword),
		NewPassword: crypto.Base64Encode(newPassword),
	}
	resp, err := c.Service.UpdateCredential(ctx, req)
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
	if c.Service == nil {
		return ErrClientNotReady
	}
	req := &server.DeleteCredentialRequest{
		Username: username,
	}
	resp, err := c.Service.DeleteCredential(ctx, req)
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
	if c.Service == nil {
		return nil, ErrClientNotReady
	}
	req := &server.ListCredUsersRequest{}
	resp, err := c.Service.ListCredUsers(ctx, req)
	if err != nil {
		return nil, err
	}
	err = handleRespStatus(resp.Status)
	if err != nil {
		return nil, err
	}
	return resp.Usernames, nil
}
