package client

import (
	"context"
	"errors"
	"testing"

	"github.com/golang/protobuf/proto"
	common "github.com/milvus-io/milvus-proto/go-api/commonpb"
	server "github.com/milvus-io/milvus-proto/go-api/milvuspb"
	"github.com/stretchr/testify/assert"
)

func TestGrpcClient_CreateCredential(t *testing.T) {
	ctx := context.Background()
	c := testClient(ctx, t)

	t.Run("create credential normal", func(t *testing.T) {
		mock.SetInjection(MCreateCredential, func(ctx context.Context, _ proto.Message) (proto.Message, error) {
			s, err := SuccessStatus()
			return s, err
		})
		defer mock.DelInjection(MCreateCredential)
		err := c.CreateCredential(ctx, testUsername, testPassword)
		assert.Nil(t, err)
	})

	t.Run("create credential invalid name", func(t *testing.T) {
		mock.SetInjection(MCreateCredential, func(ctx context.Context, _ proto.Message) (proto.Message, error) {
			s, err := BadRequestStatus()
			return s, err
		})
		defer mock.DelInjection(MCreateCredential)
		err := c.CreateCredential(ctx, "123", testPassword)
		assert.Error(t, err)
	})

	t.Run("create credential grpc error", func(t *testing.T) {
		mock.SetInjection(MCreateCredential, func(ctx context.Context, raw proto.Message) (proto.Message, error) {
			return &common.Status{}, errors.New("mocked grpc error")
		})
		defer mock.DelInjection(MCreateCredential)
		err := c.CreateCredential(ctx, testUsername, testPassword)
		assert.Error(t, err)
	})

	t.Run("create credential server error", func(t *testing.T) {
		mock.SetInjection(MCreateCredential, func(ctx context.Context, raw proto.Message) (proto.Message, error) {
			return &common.Status{
				ErrorCode: common.ErrorCode_UnexpectedError,
				Reason:    "Service is not healthy",
			}, nil
		})
		defer mock.DelInjection(MCreateCredential)
		err := c.CreateCredential(ctx, testUsername, testPassword)
		assert.Error(t, err)
	})
}

func TestGrpcClient_UpdateCredential(t *testing.T) {
	ctx := context.Background()
	c := testClient(ctx, t)

	t.Run("update credential normal", func(t *testing.T) {
		mock.SetInjection(MUpdateCredential, func(ctx context.Context, _ proto.Message) (proto.Message, error) {
			s, err := SuccessStatus()
			return s, err
		})
		defer mock.DelInjection(MUpdateCredential)
		err := c.UpdateCredential(ctx, testUsername, testPassword, testPassword)
		assert.Nil(t, err)
	})

	t.Run("update credential grpc error", func(t *testing.T) {
		mock.SetInjection(MUpdateCredential, func(ctx context.Context, raw proto.Message) (proto.Message, error) {
			return &common.Status{}, errors.New("mocked grpc error")
		})
		defer mock.DelInjection(MUpdateCredential)
		err := c.UpdateCredential(ctx, testUsername, testPassword, testPassword)
		assert.Error(t, err)
	})

	t.Run("update credential server error", func(t *testing.T) {
		mock.SetInjection(MUpdateCredential, func(ctx context.Context, raw proto.Message) (proto.Message, error) {
			return &common.Status{
				ErrorCode: common.ErrorCode_UnexpectedError,
				Reason:    "Service is not healthy",
			}, nil
		})
		defer mock.DelInjection(MUpdateCredential)
		err := c.UpdateCredential(ctx, testUsername, testPassword, testPassword)
		assert.Error(t, err)
	})
}

func TestGrpcClient_DeleteCredential(t *testing.T) {
	ctx := context.Background()
	c := testClient(ctx, t)

	t.Run("delete credential normal", func(t *testing.T) {
		mock.SetInjection(MDeleteCredential, func(ctx context.Context, _ proto.Message) (proto.Message, error) {
			s, err := SuccessStatus()
			return s, err
		})
		defer mock.DelInjection(MDeleteCredential)
		err := c.DeleteCredential(ctx, testUsername)
		assert.Nil(t, err)
	})

	t.Run("delete credential grpc error", func(t *testing.T) {
		mock.SetInjection(MDeleteCredential, func(ctx context.Context, raw proto.Message) (proto.Message, error) {
			return &common.Status{}, errors.New("mocked grpc error")
		})
		defer mock.DelInjection(MDeleteCredential)
		err := c.DeleteCredential(ctx, testUsername)
		assert.Error(t, err)
	})

	t.Run("delete credential server error", func(t *testing.T) {
		mock.SetInjection(MDeleteCredential, func(ctx context.Context, raw proto.Message) (proto.Message, error) {
			return &common.Status{
				ErrorCode: common.ErrorCode_UnexpectedError,
				Reason:    "Service is not healthy",
			}, nil
		})
		defer mock.DelInjection(MDeleteCredential)
		err := c.DeleteCredential(ctx, testUsername)
		assert.Error(t, err)
	})
}

func TestGrpcClient_ListCredUsers(t *testing.T) {
	ctx := context.Background()
	c := testClient(ctx, t)

	t.Run("list credential users normal", func(t *testing.T) {
		mock.SetInjection(MListCredUsers, func(ctx context.Context, _ proto.Message) (proto.Message, error) {
			resp := &server.ListCredUsersResponse{
				Usernames: []string{testUsername},
			}
			s, err := SuccessStatus()
			resp.Status = s
			return resp, err
		})
		defer mock.DelInjection(MListCredUsers)
		names, err := c.ListCredUsers(ctx)
		assert.Nil(t, err)
		assert.Equal(t, []string{testUsername}, names)
	})

	t.Run("list credential users grpc error", func(t *testing.T) {
		mock.SetInjection(MListCredUsers, func(ctx context.Context, raw proto.Message) (proto.Message, error) {
			return &server.ListCredUsersResponse{}, errors.New("mocked grpc error")
		})
		defer mock.DelInjection(MListCredUsers)
		_, err := c.ListCredUsers(ctx)
		assert.Error(t, err)
	})

	t.Run("list credential users server error", func(t *testing.T) {
		mock.SetInjection(MListCredUsers, func(ctx context.Context, raw proto.Message) (proto.Message, error) {
			return &server.ListCredUsersResponse{
				Status: &common.Status{
					ErrorCode: common.ErrorCode_UnexpectedError,
					Reason:    "Service is not healthy",
				},
			}, nil
		})
		defer mock.DelInjection(MListCredUsers)
		_, err := c.ListCredUsers(ctx)
		assert.Error(t, err)
	})
}
