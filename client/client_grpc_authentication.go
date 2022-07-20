package client

import (
	"context"

	"github.com/milvus-io/milvus-sdk-go/v2/internal/proto/server"
	"github.com/milvus-io/milvus-sdk-go/v2/internal/utils/crypto"
)

// CreateCredential create new user and password
func (c *grpcClient) CreateCredential(ctx context.Context, username string, password string) error {
	if c.service == nil {
		return ErrClientNotReady
	}
	req := &server.CreateCredentialRequest{
		Username: username,
		Password: crypto.Base64Encode(password),
	}
	resp, err := c.service.CreateCredential(ctx, req)
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
func (c *grpcClient) UpdateCredential(ctx context.Context, username string, oldPassword string, newPassword string) error {
	if c.service == nil {
		return ErrClientNotReady
	}
	req := &server.UpdateCredentialRequest{
		Username:    username,
		OldPassword: crypto.Base64Encode(oldPassword),
		NewPassword: crypto.Base64Encode(newPassword),
	}
	resp, err := c.service.UpdateCredential(ctx, req)
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
func (c *grpcClient) DeleteCredential(ctx context.Context, username string) error {
	if c.service == nil {
		return ErrClientNotReady
	}
	req := &server.DeleteCredentialRequest{
		Username: username,
	}
	resp, err := c.service.DeleteCredential(ctx, req)
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
func (c *grpcClient) ListCredUsers(ctx context.Context) ([]string, error) {
	if c.service == nil {
		return nil, ErrClientNotReady
	}
	req := &server.ListCredUsersRequest{}
	resp, err := c.service.ListCredUsers(ctx, req)
	if err != nil {
		return nil, err
	}
	err = handleRespStatus(resp.Status)
	if err != nil {
		return nil, err
	}
	return resp.Usernames, nil
}
