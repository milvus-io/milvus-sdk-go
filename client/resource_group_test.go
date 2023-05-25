// Copyright (C) 2019-2021 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License.

package client

import (
	"context"
	"errors"
	"fmt"
	"math/rand"
	"testing"

	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/suite"

	common "github.com/milvus-io/milvus-proto/go-api/commonpb"
	server "github.com/milvus-io/milvus-proto/go-api/milvuspb"
)

type ResourceGroupSuite struct {
	MockSuiteBase
}

func (s *ResourceGroupSuite) TestListResourceGroups() {
	c := s.client
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	s.Run("normal_run", func() {
		defer s.resetMock()

		rgs := make([]string, 0, 5)
		for i := 0; i < 5; i++ {
			rgs = append(rgs, randStr(10))
		}

		s.mock.EXPECT().ListResourceGroups(mock.Anything, mock.AnythingOfType("*milvuspb.ListResourceGroupsRequest")).
			Return(&server.ListResourceGroupsResponse{Status: &common.Status{}, ResourceGroups: rgs}, nil)

		result, err := c.ListResourceGroups(ctx)
		s.NoError(err)
		s.ElementsMatch(rgs, result)
	})

	s.Run("request_fails", func() {
		defer s.resetMock()

		s.mock.EXPECT().ListResourceGroups(mock.Anything, mock.AnythingOfType("*milvuspb.ListResourceGroupsRequest")).
			Return(nil, errors.New("mocked grpc error"))

		_, err := c.ListResourceGroups(ctx)
		s.Error(err)
	})

	s.Run("server_return_err", func() {
		defer s.resetMock()

		s.mock.EXPECT().ListResourceGroups(mock.Anything, mock.AnythingOfType("*milvuspb.ListResourceGroupsRequest")).
			Return(&server.ListResourceGroupsResponse{Status: &common.Status{ErrorCode: common.ErrorCode_UnexpectedError}}, nil)

		_, err := c.ListResourceGroups(ctx)
		s.Error(err)
	})
}

func (s *ResourceGroupSuite) TestCreateResourceGroup() {
	c := s.client
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	s.Run("normal_run", func() {
		defer s.resetMock()
		rgName := randStr(10)

		s.mock.EXPECT().CreateResourceGroup(mock.Anything, mock.AnythingOfType("*milvuspb.CreateResourceGroupRequest")).
			Run(func(_ context.Context, req *server.CreateResourceGroupRequest) {
				s.Equal(rgName, req.GetResourceGroup())
			}).
			Return(&common.Status{}, nil)

		err := c.CreateResourceGroup(ctx, rgName)
		s.NoError(err)
	})

	s.Run("request_fails", func() {
		defer s.resetMock()

		rgName := randStr(10)

		s.mock.EXPECT().CreateResourceGroup(mock.Anything, mock.AnythingOfType("*milvuspb.CreateResourceGroupRequest")).
			Run(func(_ context.Context, req *server.CreateResourceGroupRequest) {
				s.Equal(rgName, req.GetResourceGroup())
			}).
			Return(nil, errors.New("mocked grpc error"))

		err := c.CreateResourceGroup(ctx, rgName)
		s.Error(err)
	})

	s.Run("server_return_err", func() {
		defer s.resetMock()

		rgName := randStr(10)

		s.mock.EXPECT().CreateResourceGroup(mock.Anything, mock.AnythingOfType("*milvuspb.CreateResourceGroupRequest")).
			Run(func(_ context.Context, req *server.CreateResourceGroupRequest) {
				s.Equal(rgName, req.GetResourceGroup())
			}).
			Return(&common.Status{ErrorCode: common.ErrorCode_UnexpectedError}, nil)

		err := c.CreateResourceGroup(ctx, rgName)
		s.Error(err)
	})
}

func (s *ResourceGroupSuite) TestDescribeResourceGroup() {
	c := s.client
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	s.Run("normal_run", func() {
		defer s.resetMock()
		rgName := randStr(10)
		capacity := rand.Int31n(5) + 1
		available := rand.Int31n(capacity)

		loadedReplica := make(map[string]int32)
		l := rand.Intn(3) + 1
		for i := 0; i < l; i++ {
			loadedReplica[fmt.Sprintf("coll_%s", randStr(6))] = rand.Int31n(5) + 1
		}
		incomingNum := make(map[string]int32)
		l = rand.Intn(3) + 1
		for i := 0; i < l; i++ {
			incomingNum[fmt.Sprintf("coll_%s", randStr(6))] = rand.Int31n(5) + 1
		}
		outgoingNum := make(map[string]int32)
		l = rand.Intn(3) + 1
		for i := 0; i < l; i++ {
			outgoingNum[fmt.Sprintf("coll_%s", randStr(6))] = rand.Int31n(5) + 1
		}

		s.mock.EXPECT().DescribeResourceGroup(mock.Anything, mock.AnythingOfType("*milvuspb.DescribeResourceGroupRequest")).
			Run(func(_ context.Context, req *server.DescribeResourceGroupRequest) {
				s.Equal(rgName, req.GetResourceGroup())
			}).
			Call.Return(func(_ context.Context, req *server.DescribeResourceGroupRequest) *server.DescribeResourceGroupResponse {

			return &server.DescribeResourceGroupResponse{
				Status: &common.Status{},
				ResourceGroup: &server.ResourceGroup{
					Name:             rgName,
					Capacity:         capacity,
					NumAvailableNode: available,
					NumLoadedReplica: loadedReplica,
					NumOutgoingNode:  outgoingNum,
					NumIncomingNode:  incomingNum,
				},
			}
		}, nil)

		result, err := c.DescribeResourceGroup(ctx, rgName)
		s.NoError(err)
		s.Equal(rgName, result.Name)
		s.Equal(capacity, result.Capacity)
		s.Equal(available, result.AvailableNodesNumber)
		s.InDeltaMapValues(loadedReplica, result.LoadedReplica, 0)
		s.InDeltaMapValues(incomingNum, result.IncomingNodeNum, 0)
		s.InDeltaMapValues(outgoingNum, result.OutgoingNodeNum, 0)
	})

	s.Run("request_fails", func() {
		defer s.resetMock()

		rgName := randStr(10)

		s.mock.EXPECT().DescribeResourceGroup(mock.Anything, mock.AnythingOfType("*milvuspb.DescribeResourceGroupRequest")).
			Run(func(_ context.Context, req *server.DescribeResourceGroupRequest) {
				s.Equal(rgName, req.GetResourceGroup())
			}).
			Return(nil, errors.New("mocked grpc error"))

		_, err := c.DescribeResourceGroup(ctx, rgName)
		s.Error(err)
	})

	s.Run("server_return_err", func() {
		defer s.resetMock()

		rgName := randStr(10)

		s.mock.EXPECT().DescribeResourceGroup(mock.Anything, mock.AnythingOfType("*milvuspb.DescribeResourceGroupRequest")).
			Run(func(_ context.Context, req *server.DescribeResourceGroupRequest) {
				s.Equal(rgName, req.GetResourceGroup())
			}).
			Return(&server.DescribeResourceGroupResponse{Status: &common.Status{ErrorCode: common.ErrorCode_UnexpectedError}}, nil)

		_, err := c.DescribeResourceGroup(ctx, rgName)
		s.Error(err)
	})
}

func (s *ResourceGroupSuite) TestDropResourceGroup() {
	c := s.client
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	s.Run("normal_run", func() {
		defer s.resetMock()
		rgName := randStr(10)

		s.mock.EXPECT().DropResourceGroup(mock.Anything, mock.AnythingOfType("*milvuspb.DropResourceGroupRequest")).
			Run(func(_ context.Context, req *server.DropResourceGroupRequest) {
				s.Equal(rgName, req.GetResourceGroup())
			}).
			Return(&common.Status{}, nil)

		err := c.DropResourceGroup(ctx, rgName)
		s.NoError(err)
	})

	s.Run("request_fails", func() {
		defer s.resetMock()

		rgName := randStr(10)

		s.mock.EXPECT().DropResourceGroup(mock.Anything, mock.AnythingOfType("*milvuspb.DropResourceGroupRequest")).
			Run(func(_ context.Context, req *server.DropResourceGroupRequest) {
				s.Equal(rgName, req.GetResourceGroup())
			}).
			Return(nil, errors.New("mocked grpc error"))

		err := c.DropResourceGroup(ctx, rgName)
		s.Error(err)
	})

	s.Run("server_return_err", func() {
		defer s.resetMock()

		rgName := randStr(10)

		s.mock.EXPECT().DropResourceGroup(mock.Anything, mock.AnythingOfType("*milvuspb.DropResourceGroupRequest")).
			Run(func(_ context.Context, req *server.DropResourceGroupRequest) {
				s.Equal(rgName, req.GetResourceGroup())
			}).
			Return(&common.Status{ErrorCode: common.ErrorCode_UnexpectedError}, nil)

		err := c.DropResourceGroup(ctx, rgName)
		s.Error(err)
	})
}

func (s *ResourceGroupSuite) TestTransferNodes() {
	c := s.client
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	s.Run("normal_run", func() {
		defer s.resetMock()
		sourceRg := randStr(10)
		targetRg := randStr(10)
		nodeNum := rand.Int31n(5) + 1

		s.mock.EXPECT().TransferNode(mock.Anything, mock.AnythingOfType("*milvuspb.TransferNodeRequest")).
			Run(func(_ context.Context, req *server.TransferNodeRequest) {
				s.Equal(sourceRg, req.GetSourceResourceGroup())
				s.Equal(targetRg, req.GetTargetResourceGroup())
				s.Equal(nodeNum, req.GetNumNode())
			}).
			Return(&common.Status{}, nil)

		err := c.TransferNode(ctx, sourceRg, targetRg, nodeNum)
		s.NoError(err)
	})

	s.Run("request_fails", func() {
		defer s.resetMock()

		sourceRg := randStr(10)
		targetRg := randStr(10)
		nodeNum := rand.Int31n(5) + 1

		s.mock.EXPECT().TransferNode(mock.Anything, mock.AnythingOfType("*milvuspb.TransferNodeRequest")).
			Run(func(_ context.Context, req *server.TransferNodeRequest) {
				s.Equal(sourceRg, req.GetSourceResourceGroup())
				s.Equal(targetRg, req.GetTargetResourceGroup())
				s.Equal(nodeNum, req.GetNumNode())
			}).
			Return(nil, errors.New("mocked grpc error"))

		err := c.TransferNode(ctx, sourceRg, targetRg, nodeNum)
		s.Error(err)
	})

	s.Run("server_return_err", func() {
		defer s.resetMock()

		sourceRg := randStr(10)
		targetRg := randStr(10)
		nodeNum := rand.Int31n(5) + 1

		s.mock.EXPECT().TransferNode(mock.Anything, mock.AnythingOfType("*milvuspb.TransferNodeRequest")).
			Run(func(_ context.Context, req *server.TransferNodeRequest) {
				s.Equal(sourceRg, req.GetSourceResourceGroup())
				s.Equal(targetRg, req.GetTargetResourceGroup())
				s.Equal(nodeNum, req.GetNumNode())
			}).
			Return(&common.Status{ErrorCode: common.ErrorCode_UnexpectedError}, nil)

		err := c.TransferNode(ctx, sourceRg, targetRg, nodeNum)
		s.Error(err)
	})
}

func (s *ResourceGroupSuite) TestTransferReplica() {
	c := s.client
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	s.Run("normal_run", func() {
		defer s.resetMock()
		collName := fmt.Sprintf("coll_%s", randStr(6))
		sourceRg := randStr(10)
		targetRg := randStr(10)
		replicaNum := rand.Int63n(5) + 1

		s.mock.EXPECT().TransferReplica(mock.Anything, mock.AnythingOfType("*milvuspb.TransferReplicaRequest")).
			Run(func(_ context.Context, req *server.TransferReplicaRequest) {
				s.Equal(sourceRg, req.GetSourceResourceGroup())
				s.Equal(targetRg, req.GetTargetResourceGroup())
				s.Equal(replicaNum, req.GetNumReplica())
			}).
			Return(&common.Status{}, nil)

		err := c.TransferReplica(ctx, sourceRg, targetRg, collName, replicaNum)
		s.NoError(err)
	})

	s.Run("request_fails", func() {
		defer s.resetMock()

		collName := fmt.Sprintf("coll_%s", randStr(6))
		sourceRg := randStr(10)
		targetRg := randStr(10)
		replicaNum := rand.Int63n(5) + 1

		s.mock.EXPECT().TransferReplica(mock.Anything, mock.AnythingOfType("*milvuspb.TransferReplicaRequest")).
			Run(func(_ context.Context, req *server.TransferReplicaRequest) {
				s.Equal(sourceRg, req.GetSourceResourceGroup())
				s.Equal(targetRg, req.GetTargetResourceGroup())
				s.Equal(replicaNum, req.GetNumReplica())
			}).
			Return(nil, errors.New("mocked grpc error"))

		err := c.TransferReplica(ctx, sourceRg, targetRg, collName, replicaNum)
		s.Error(err)
	})

	s.Run("server_return_err", func() {
		defer s.resetMock()

		collName := fmt.Sprintf("coll_%s", randStr(6))
		sourceRg := randStr(10)
		targetRg := randStr(10)
		replicaNum := rand.Int63n(5) + 1

		s.mock.EXPECT().TransferReplica(mock.Anything, mock.AnythingOfType("*milvuspb.TransferReplicaRequest")).
			Run(func(_ context.Context, req *server.TransferReplicaRequest) {
				s.Equal(sourceRg, req.GetSourceResourceGroup())
				s.Equal(targetRg, req.GetTargetResourceGroup())
				s.Equal(replicaNum, req.GetNumReplica())
			}).
			Return(&common.Status{ErrorCode: common.ErrorCode_UnexpectedError}, nil)

		err := c.TransferReplica(ctx, sourceRg, targetRg, collName, replicaNum)
		s.Error(err)
	})
}

func TestResourceGroupSuite(t *testing.T) {
	suite.Run(t, new(ResourceGroupSuite))
}
