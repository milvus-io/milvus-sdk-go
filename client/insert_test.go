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
	"testing"

	"github.com/cockroachdb/errors"
	"github.com/milvus-io/milvus-proto/go-api/v2/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v2/milvuspb"
	"github.com/milvus-io/milvus-proto/go-api/v2/schemapb"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/suite"
)

type InsertSuite struct {
	MockSuiteBase
}

func (s *InsertSuite) TestInsertFail() {
	c := s.client
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	s.Run("collection_not_exist", func() {
		defer s.resetMock()
		s.setupHasCollection()
		_, err := c.Insert(ctx, testCollectionName, "")
		s.Error(err)
	})

	s.Run("partition_not_exist", func() {
		defer s.resetMock()
		s.setupHasCollection(testCollectionName)
		s.setupHasPartition(testCollectionName)

		_, err := c.Insert(ctx, testCollectionName, "partition_name")
		s.Error(err)
	})

	s.Run("field_not_exist", func() {
		defer s.resetMock()
		s.setupHasCollection(testCollectionName)
		s.setupHasPartition(testCollectionName, "partition_1")
		s.setupDescribeCollection(testCollectionName, entity.NewSchema().
			WithField(entity.NewField().WithIsPrimaryKey(true).WithIsAutoID(true).WithName("ID").WithDataType(entity.FieldTypeInt64)).
			WithField(entity.NewField().WithName("vector").WithDataType(entity.FieldTypeFloatVector).WithTypeParams(entity.TypeParamDim, "128")),
		)

		_, err := c.Insert(ctx, testCollectionName, "partition_1",
			entity.NewColumnInt64("extra", []int64{1}),
		)
		s.Error(err)
	})

	s.Run("missing_field_without_default_value", func() {
		defer s.resetMock()
		s.setupHasCollection(testCollectionName)
		s.setupHasPartition(testCollectionName, "partition_1")

		s.setupDescribeCollection(testCollectionName, entity.NewSchema().
			WithField(entity.NewField().WithIsPrimaryKey(true).WithIsAutoID(true).WithName("ID").WithDataType(entity.FieldTypeInt64)).
			WithField(entity.NewField().WithName("vector").WithDataType(entity.FieldTypeFloatVector).WithTypeParams(entity.TypeParamDim, "128")),
		)

		_, err := c.Insert(ctx, testCollectionName, "partition_1",
			entity.NewColumnInt64("ID", []int64{1}),
		)
		s.Error(err)
	})

	s.Run("column_len_not_match", func() {
		defer s.resetMock()
		s.setupHasCollection(testCollectionName)
		s.setupHasPartition(testCollectionName, "partition_1")

		s.setupDescribeCollection(testCollectionName, entity.NewSchema().
			WithField(entity.NewField().WithIsPrimaryKey(true).WithIsAutoID(true).WithName("ID").WithDataType(entity.FieldTypeInt64)).
			WithField(entity.NewField().WithName("vector").WithDataType(entity.FieldTypeFloatVector).WithTypeParams(entity.TypeParamDim, "128")),
		)

		_, err := c.Insert(ctx, testCollectionName, "partition_1",
			entity.NewColumnInt64("ID", []int64{1, 2}),
			entity.NewColumnFloatVector("vector", 128, [][]float32{}),
		)
		s.Error(err)
	})

	s.Run("duplicated column", func() {
		defer s.resetMock()
		s.setupHasCollection(testCollectionName)
		s.setupHasPartition(testCollectionName, "partition_1")

		s.setupDescribeCollection(testCollectionName, entity.NewSchema().
			WithField(entity.NewField().WithIsPrimaryKey(true).WithIsAutoID(true).WithName("ID").WithDataType(entity.FieldTypeInt64)).
			WithField(entity.NewField().WithName("vector").WithDataType(entity.FieldTypeFloatVector).WithTypeParams(entity.TypeParamDim, "128")),
		)

		s.mock.EXPECT().Insert(mock.Anything, mock.AnythingOfType("*milvuspb.InsertRequest")).
			Run(func(ctx context.Context, req *milvuspb.InsertRequest) {
				s.Equal(1, len(req.GetFieldsData()))
			}).Return(&milvuspb.MutationResult{
			Status: &commonpb.Status{},
			IDs: &schemapb.IDs{
				IdField: &schemapb.IDs_IntId{
					IntId: &schemapb.LongArray{
						Data: []int64{1},
					},
				},
			},
		}, nil)

		_, err := c.Insert(ctx, testCollectionName, "partition_1",
			entity.NewColumnFloatVector("vector", 128, generateFloatVector(1, 128)),
			entity.NewColumnFloatVector("vector", 128, generateFloatVector(1, 128)),
		)

		s.Error(err)
	})

	s.Run("dim_not_match", func() {
		defer s.resetMock()
		s.setupHasCollection(testCollectionName)
		s.setupHasPartition(testCollectionName, "partition_1")

		s.setupDescribeCollection(testCollectionName, entity.NewSchema().
			WithField(entity.NewField().WithIsPrimaryKey(true).WithIsAutoID(true).WithName("ID").WithDataType(entity.FieldTypeInt64)).
			WithField(entity.NewField().WithName("vector").WithDataType(entity.FieldTypeFloatVector).WithTypeParams(entity.TypeParamDim, "128")),
		)

		_, err := c.Insert(ctx, testCollectionName, "partition_1",
			entity.NewColumnInt64("ID", []int64{1}),
			entity.NewColumnFloatVector("vector", 8, [][]float32{{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}}),
		)
		s.Error(err)
	})

	s.Run("server_insert_fail", func() {
		defer s.resetMock()
		s.setupHasCollection(testCollectionName)
		s.setupHasPartition(testCollectionName, "partition_1")

		s.setupDescribeCollection(testCollectionName, entity.NewSchema().
			WithField(entity.NewField().WithIsPrimaryKey(true).WithIsAutoID(true).WithName("ID").WithDataType(entity.FieldTypeInt64)).
			WithField(entity.NewField().WithName("vector").WithDataType(entity.FieldTypeFloatVector).WithTypeParams(entity.TypeParamDim, "128")),
		)

		s.mock.EXPECT().Insert(mock.Anything, mock.AnythingOfType("*milvuspb.InsertRequest")).Return(
			&milvuspb.MutationResult{Status: &commonpb.Status{ErrorCode: commonpb.ErrorCode_UnexpectedError}}, nil,
		)

		_, err := c.Insert(ctx, testCollectionName, "partition_1",
			entity.NewColumnFloatVector("vector", 128, generateFloatVector(1, 128)),
		)

		s.Error(err)
	})

	s.Run("server_connection_error", func() {
		defer s.resetMock()
		s.setupHasCollection(testCollectionName)
		s.setupHasPartition(testCollectionName, "partition_1")

		s.setupDescribeCollection(testCollectionName, entity.NewSchema().
			WithField(entity.NewField().WithIsPrimaryKey(true).WithIsAutoID(true).WithName("ID").WithDataType(entity.FieldTypeInt64)).
			WithField(entity.NewField().WithName("vector").WithDataType(entity.FieldTypeFloatVector).WithTypeParams(entity.TypeParamDim, "128")),
		)

		s.mock.EXPECT().Insert(mock.Anything, mock.AnythingOfType("*milvuspb.InsertRequest")).Return(
			nil, errors.New("mocked error"),
		)

		_, err := c.Insert(ctx, testCollectionName, "partition_1",
			entity.NewColumnFloatVector("vector", 128, generateFloatVector(1, 128)),
		)

		s.Error(err)
	})
}

func (s *InsertSuite) TestInsertSuccess() {
	c := s.client
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	s.Run("non_dynamic_schema", func() {
		defer s.resetMock()
		s.setupHasCollection(testCollectionName)
		s.setupHasPartition(testCollectionName, "partition_1")

		s.setupDescribeCollection(testCollectionName, entity.NewSchema().
			WithField(entity.NewField().WithIsPrimaryKey(true).WithIsAutoID(true).WithName("ID").WithDataType(entity.FieldTypeInt64)).
			WithField(entity.NewField().WithName("vector").WithDataType(entity.FieldTypeFloatVector).WithTypeParams(entity.TypeParamDim, "128")),
		)

		s.mock.EXPECT().Insert(mock.Anything, mock.AnythingOfType("*milvuspb.InsertRequest")).
			Run(func(ctx context.Context, req *milvuspb.InsertRequest) {
				s.Equal(1, len(req.GetFieldsData()))
			}).Return(&milvuspb.MutationResult{
			Status: &commonpb.Status{},
			IDs: &schemapb.IDs{
				IdField: &schemapb.IDs_IntId{
					IntId: &schemapb.LongArray{
						Data: []int64{1},
					},
				},
			},
		}, nil)

		r, err := c.Insert(ctx, testCollectionName, "partition_1",
			entity.NewColumnFloatVector("vector", 128, generateFloatVector(1, 128)),
		)

		s.NoError(err)
		s.Equal(1, r.Len())
	})

	s.Run("dynamic_field_schema", func() {
		defer s.resetMock()
		s.setupHasCollection(testCollectionName)
		s.setupHasPartition(testCollectionName, "partition_1")

		s.setupDescribeCollection(testCollectionName, entity.NewSchema().
			WithDynamicFieldEnabled(true).
			WithField(entity.NewField().WithIsPrimaryKey(true).WithIsAutoID(true).WithName("ID").WithDataType(entity.FieldTypeInt64)).
			WithField(entity.NewField().WithName("vector").WithDataType(entity.FieldTypeFloatVector).WithTypeParams(entity.TypeParamDim, "128")),
		)

		s.mock.EXPECT().Insert(mock.Anything, mock.AnythingOfType("*milvuspb.InsertRequest")).
			Run(func(ctx context.Context, req *milvuspb.InsertRequest) {
				s.Equal(2, len(req.GetFieldsData()))
				var found bool
				for _, fd := range req.GetFieldsData() {
					if fd.GetFieldName() == "" && fd.GetIsDynamic() {
						found = true
						break
					}
				}
				s.True(found)
			}).Return(&milvuspb.MutationResult{
			Status: &commonpb.Status{},
			IDs: &schemapb.IDs{
				IdField: &schemapb.IDs_IntId{
					IntId: &schemapb.LongArray{
						Data: []int64{1},
					},
				},
			},
		}, nil)

		r, err := c.Insert(ctx, testCollectionName, "partition_1",
			entity.NewColumnInt64("extra", []int64{1}),
			entity.NewColumnFloatVector("vector", 128, generateFloatVector(1, 128)),
		)

		s.NoError(err)
		s.Equal(1, r.Len())
	})

	s.Run("missing_field_with_default_value", func() {
		s.T().Skip("skip for default value test")
		defer s.resetMock()
		s.setupHasCollection(testCollectionName)
		s.setupHasPartition(testCollectionName, "partition_1")

		s.setupDescribeCollection(testCollectionName, entity.NewSchema().
			WithField(entity.NewField().WithIsPrimaryKey(true).WithIsAutoID(true).WithName("ID").WithDataType(entity.FieldTypeInt64)).
			WithField(entity.NewField().WithName("default_value").WithDataType(entity.FieldTypeInt64)).
			WithField(entity.NewField().WithName("vector").WithDataType(entity.FieldTypeFloatVector).WithTypeParams(entity.TypeParamDim, "128")),
		)

		s.mock.EXPECT().Insert(mock.Anything, mock.AnythingOfType("*milvuspb.InsertRequest")).
			Run(func(ctx context.Context, req *milvuspb.InsertRequest) {
				s.Equal(2, len(req.GetFieldsData()))
			}).Return(&milvuspb.MutationResult{
			Status: &commonpb.Status{},
			IDs: &schemapb.IDs{
				IdField: &schemapb.IDs_IntId{
					IntId: &schemapb.LongArray{
						Data: []int64{1},
					},
				},
			},
		}, nil)

		r, err := c.Insert(ctx, testCollectionName, "partition_1",
			entity.NewColumnFloatVector("vector", 128, generateFloatVector(1, 128)),
		)
		s.NoError(err)
		s.Equal(1, r.Len())
	})
}

func TestGrpcInsert(t *testing.T) {
	suite.Run(t, new(InsertSuite))
}
