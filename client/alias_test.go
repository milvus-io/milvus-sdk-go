// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package client

import (
	"context"
	"fmt"
	"testing"

	"github.com/cockroachdb/errors"
	"github.com/milvus-io/milvus-proto/go-api/v2/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v2/milvuspb"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/suite"
)

type AliasSuite struct {
	MockSuiteBase
}

func (s *AliasSuite) TestCreateAlias() {
	c := s.client
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	s.Run("normal_create", func() {
		defer s.resetMock()

		collName := fmt.Sprintf("coll_%s", randStr(6))
		alias := fmt.Sprintf("alias_%s", randStr(6))

		s.mock.EXPECT().CreateAlias(mock.Anything, mock.AnythingOfType("*milvuspb.CreateAliasRequest")).
			Run(func(ctx context.Context, req *milvuspb.CreateAliasRequest) {
				s.Equal(collName, req.GetCollectionName())
				s.Equal(alias, req.GetAlias())
			}).Return(s.getSuccessStatus(), nil)
		err := c.CreateAlias(ctx, collName, alias)
		s.NoError(err)
	})

	s.Run("failure_cases", func() {
		collName := fmt.Sprintf("coll_%s", randStr(6))
		alias := fmt.Sprintf("alias_%s", randStr(6))

		s.Run("return_error", func() {
			defer s.resetMock()
			s.mock.EXPECT().CreateAlias(mock.Anything, mock.AnythingOfType("*milvuspb.CreateAliasRequest")).
				Return(nil, errors.New("mocked"))
			err := c.CreateAlias(ctx, collName, alias)
			s.Error(err)
		})

		s.Run("failure_status", func() {
			defer s.resetMock()

			s.mock.EXPECT().CreateAlias(mock.Anything, mock.AnythingOfType("*milvuspb.CreateAliasRequest")).
				Return(s.getStatus(commonpb.ErrorCode_UnexpectedError, "mocked"), nil)
			err := c.CreateAlias(ctx, collName, alias)
			s.Error(err)
		})
	})

	s.Run("invalid_client", func() {
		c := &GrpcClient{}
		collName := fmt.Sprintf("coll_%s", randStr(6))
		alias := fmt.Sprintf("alias_%s", randStr(6))

		err := c.CreateAlias(ctx, collName, alias)
		s.Error(err)
	})
}

func (s *AliasSuite) TestDropAlias() {
	c := s.client
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	s.Run("normal_create", func() {
		defer s.resetMock()

		alias := fmt.Sprintf("alias_%s", randStr(6))

		s.mock.EXPECT().DropAlias(mock.Anything, mock.AnythingOfType("*milvuspb.DropAliasRequest")).
			Run(func(ctx context.Context, req *milvuspb.DropAliasRequest) {
				s.Equal(alias, req.GetAlias())
			}).Return(s.getSuccessStatus(), nil)
		err := c.DropAlias(ctx, alias)
		s.NoError(err)
	})

	s.Run("failure_cases", func() {
		alias := fmt.Sprintf("alias_%s", randStr(6))
		s.Run("return_error", func() {
			defer s.resetMock()
			s.mock.EXPECT().DropAlias(mock.Anything, mock.AnythingOfType("*milvuspb.DropAliasRequest")).
				Return(nil, errors.New("mocked"))
			err := c.DropAlias(ctx, alias)
			s.Error(err)
		})

		s.Run("failure_status", func() {
			defer s.resetMock()

			s.mock.EXPECT().DropAlias(mock.Anything, mock.AnythingOfType("*milvuspb.DropAliasRequest")).
				Return(s.getStatus(commonpb.ErrorCode_UnexpectedError, "mocked"), nil)
			err := c.DropAlias(ctx, alias)
			s.Error(err)
		})
	})

	s.Run("invalid_client", func() {
		c := &GrpcClient{}
		alias := fmt.Sprintf("alias_%s", randStr(6))

		err := c.DropAlias(ctx, alias)
		s.Error(err)
	})
}

func (s *AliasSuite) TestAlterAlias() {
	c := s.client
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	s.Run("normal_create", func() {
		defer s.resetMock()

		collName := fmt.Sprintf("coll_%s", randStr(6))
		alias := fmt.Sprintf("alias_%s", randStr(6))

		s.mock.EXPECT().AlterAlias(mock.Anything, mock.AnythingOfType("*milvuspb.AlterAliasRequest")).
			Run(func(ctx context.Context, req *milvuspb.AlterAliasRequest) {
				s.Equal(collName, req.GetCollectionName())
				s.Equal(alias, req.GetAlias())
			}).Return(s.getSuccessStatus(), nil)
		err := c.AlterAlias(ctx, collName, alias)
		s.NoError(err)
	})

	s.Run("failure_cases", func() {
		collName := fmt.Sprintf("coll_%s", randStr(6))
		alias := fmt.Sprintf("alias_%s", randStr(6))

		s.Run("return_error", func() {
			defer s.resetMock()
			s.mock.EXPECT().AlterAlias(mock.Anything, mock.AnythingOfType("*milvuspb.AlterAliasRequest")).
				Return(nil, errors.New("mocked"))
			err := c.AlterAlias(ctx, collName, alias)
			s.Error(err)
		})

		s.Run("failure_status", func() {
			defer s.resetMock()

			s.mock.EXPECT().AlterAlias(mock.Anything, mock.AnythingOfType("*milvuspb.AlterAliasRequest")).
				Return(s.getStatus(commonpb.ErrorCode_UnexpectedError, "mocked"), nil)
			err := c.AlterAlias(ctx, collName, alias)
			s.Error(err)
		})
	})

	s.Run("invalid_client", func() {
		c := &GrpcClient{}
		collName := fmt.Sprintf("coll_%s", randStr(6))
		alias := fmt.Sprintf("alias_%s", randStr(6))

		err := c.AlterAlias(ctx, collName, alias)
		s.Error(err)
	})

}

func TestAlias(t *testing.T) {
	suite.Run(t, new(AliasSuite))
}
