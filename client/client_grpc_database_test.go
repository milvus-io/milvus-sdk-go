package client

import (
	"context"
	"testing"

	"github.com/go-faker/faker/v4"
	"github.com/go-faker/faker/v4/pkg/options"
	"github.com/golang/protobuf/proto"
	server "github.com/milvus-io/milvus-proto/go-api/milvuspb"
	"github.com/stretchr/testify/assert"
)

func TestGrpcClientListDatabases(t *testing.T) {
	ctx := context.Background()
	c := testClient(ctx, t)

	type testCase struct {
		DBName []string
	}
	caseList := []testCase{}
	faker.FakeData(&caseList, options.WithRandomMapAndSliceMaxSize(25))

	for _, singleCase := range caseList {
		mockServer.SetInjection(MListDatabase, func(ctx context.Context, m proto.Message) (proto.Message, error) {
			s, err := SuccessStatus()
			resp := &server.ListDatabasesResponse{
				Status:  s,
				DbNames: singleCase.DBName,
			}
			return resp, err
		})
		dbNames, err := c.ListDatabases(ctx)
		assert.Nil(t, err)
		assert.Equal(t, len(singleCase.DBName), len(dbNames))
		for i, db := range singleCase.DBName {
			assert.Equal(t, db, dbNames[i].Name)
		}
	}
}

func TestGrpcClientCreateDatabase(t *testing.T) {
	ctx := context.Background()
	c := testClient(ctx, t)
	mockServer.SetInjection(MCreateDatabase, func(ctx context.Context, m proto.Message) (proto.Message, error) {
		return SuccessStatus()
	})
	err := c.CreateDatabase(ctx, "a")
	assert.Nil(t, err)
}

func TestGrpcClientDropDatabase(t *testing.T) {
	ctx := context.Background()
	c := testClient(ctx, t)
	mockServer.SetInjection(MDropDatabase, func(ctx context.Context, m proto.Message) (proto.Message, error) {
		return SuccessStatus()
	})
	err := c.DropDatabase(ctx, "a")
	assert.Nil(t, err)
}
