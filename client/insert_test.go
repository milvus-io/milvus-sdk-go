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
	"fmt"
	"math/rand"
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
		s.setupDescribeCollectionError(commonpb.ErrorCode_UnexpectedError, errors.New("mocked"))
		_, err := c.Insert(ctx, testCollectionName, "")
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
		defer s.resetMock()
		s.setupHasCollection(testCollectionName)
		s.setupHasPartition(testCollectionName, "partition_1")

		s.setupDescribeCollection(testCollectionName, entity.NewSchema().
			WithField(entity.NewField().WithIsPrimaryKey(true).WithIsAutoID(true).WithName("ID").WithDataType(entity.FieldTypeInt64)).
			WithField(entity.NewField().WithName("default_value").WithDataType(entity.FieldTypeInt64).WithDefaultValueLong(1)).
			WithField(entity.NewField().WithName("vector").WithDataType(entity.FieldTypeFloatVector).WithTypeParams(entity.TypeParamDim, "128")),
		)

		s.mock.EXPECT().Insert(mock.Anything, mock.AnythingOfType("*milvuspb.InsertRequest")).
			Run(func(ctx context.Context, req *milvuspb.InsertRequest) {
				var data *schemapb.FieldData
				for _, d := range req.GetFieldsData() {
					if d.FieldName == "default_value" {
						data = d
					}
				}
				s.Equal([]bool{false}, data.ValidData)
				s.Equal(0, len(data.GetScalars().GetLongData().Data))
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

	s.Run("missing_field_with_nullable", func() {
		defer s.resetMock()
		s.setupHasCollection(testCollectionName)
		s.setupHasPartition(testCollectionName, "partition_1")

		s.setupDescribeCollection(testCollectionName, entity.NewSchema().
			WithField(entity.NewField().WithIsPrimaryKey(true).WithIsAutoID(true).WithName("ID").WithDataType(entity.FieldTypeInt64)).
			WithField(entity.NewField().WithName("nullable_fid").WithDataType(entity.FieldTypeInt64).WithNullable(true)).
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

	s.Run("insert_with_nullable_column", func() {
		defer s.resetMock()
		s.setupHasCollection(testCollectionName)
		s.setupHasPartition(testCollectionName, "partition_1")

		s.setupDescribeCollection(testCollectionName, entity.NewSchema().
			WithField(entity.NewField().WithIsPrimaryKey(true).WithIsAutoID(true).WithName("ID").WithDataType(entity.FieldTypeInt64)).
			WithField(entity.NewField().WithName("nullable_fid").WithDataType(entity.FieldTypeInt64).WithNullable(true)).
			WithField(entity.NewField().WithName("vector").WithDataType(entity.FieldTypeFloatVector).WithTypeParams(entity.TypeParamDim, "128")),
		)

		s.mock.EXPECT().Insert(mock.Anything, mock.AnythingOfType("*milvuspb.InsertRequest")).
			Run(func(ctx context.Context, req *milvuspb.InsertRequest) {
				var data *schemapb.FieldData
				for _, d := range req.GetFieldsData() {
					if d.FieldName == "nullable_fid" {
						data = d
					}
				}
				s.Equal([]bool{false, true}, data.ValidData)
				s.Equal([]int64{1}, data.GetScalars().GetLongData().Data)
				s.Equal(2, len(req.GetFieldsData()))
			}).Return(&milvuspb.MutationResult{
			Status: &commonpb.Status{},
			IDs: &schemapb.IDs{
				IdField: &schemapb.IDs_IntId{
					IntId: &schemapb.LongArray{
						Data: []int64{1, 2},
					},
				},
			},
		}, nil)
		value := make([]int64, 2)
		value[1] = 1
		validValue := make([]bool, 2)
		validValue[1] = true
		r, err := c.Insert(ctx, testCollectionName, "partition_1",
			entity.NewNullableColumnInt64("nullable_fid", value, validValue),
			entity.NewColumnFloatVector("vector", 128, generateFloatVector(2, 128)),
		)
		s.NoError(err)
		s.Equal(2, r.Len())
	})

	s.Run("insert_with_nullable_column_with_default_value", func() {
		defer s.resetMock()
		s.setupHasCollection(testCollectionName)
		s.setupHasPartition(testCollectionName, "partition_1")

		s.setupDescribeCollection(testCollectionName, entity.NewSchema().
			WithField(entity.NewField().WithIsPrimaryKey(true).WithIsAutoID(true).WithName("ID").WithDataType(entity.FieldTypeInt64)).
			WithField(entity.NewField().WithName("nullable_fid").WithDataType(entity.FieldTypeInt64).WithNullable(true).WithDefaultValueLong(10)).
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
						Data: []int64{1, 2},
					},
				},
			},
		}, nil)
		value := make([]int64, 2)
		value[1] = 1
		validValue := make([]bool, 2)
		validValue[1] = true
		r, err := c.Insert(ctx, testCollectionName, "partition_1",
			entity.NewNullableColumnInt64("nullable_fid", value, validValue),
			entity.NewColumnFloatVector("vector", 128, generateFloatVector(2, 128)),
		)
		s.NoError(err)
		s.Equal(2, r.Len())
	})
}

type UpsertSuite struct {
	MockSuiteBase
}

func (s *UpsertSuite) TestUpsertFail() {
	c := s.client
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	s.Run("collection_not_exist", func() {
		defer s.resetMock()
		s.setupDescribeCollectionError(commonpb.ErrorCode_UnexpectedError, errors.New("mocked"))
		_, err := c.Upsert(ctx, testCollectionName, "")
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

		_, err := c.Upsert(ctx, testCollectionName, "partition_1",
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

		_, err := c.Upsert(ctx, testCollectionName, "partition_1",
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

		_, err := c.Upsert(ctx, testCollectionName, "partition_1",
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

		s.mock.EXPECT().Upsert(mock.Anything, mock.AnythingOfType("*milvuspb.InsertRequest")).
			Run(func(ctx context.Context, req *milvuspb.UpsertRequest) {
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

		_, err := c.Upsert(ctx, testCollectionName, "partition_1",
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

		_, err := c.Upsert(ctx, testCollectionName, "partition_1",
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

		s.mock.EXPECT().Upsert(mock.Anything, mock.AnythingOfType("*milvuspb.UpsertRequest")).Return(
			&milvuspb.MutationResult{Status: &commonpb.Status{ErrorCode: commonpb.ErrorCode_UnexpectedError}}, nil,
		)

		_, err := c.Upsert(ctx, testCollectionName, "partition_1",
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

		s.mock.EXPECT().Upsert(mock.Anything, mock.AnythingOfType("*milvuspb.UpsertRequest")).Return(
			nil, errors.New("mocked error"),
		)

		_, err := c.Upsert(ctx, testCollectionName, "partition_1",
			entity.NewColumnFloatVector("vector", 128, generateFloatVector(1, 128)),
		)

		s.Error(err)
	})
}

func (s *UpsertSuite) TestUpsertSuccess() {
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

		s.mock.EXPECT().Upsert(mock.Anything, mock.AnythingOfType("*milvuspb.UpsertRequest")).
			Run(func(ctx context.Context, req *milvuspb.UpsertRequest) {
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

		r, err := c.Upsert(ctx, testCollectionName, "partition_1",
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

		s.mock.EXPECT().Upsert(mock.Anything, mock.AnythingOfType("*milvuspb.UpsertRequest")).
			Run(func(ctx context.Context, req *milvuspb.UpsertRequest) {
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

		r, err := c.Upsert(ctx, testCollectionName, "partition_1",
			entity.NewColumnInt64("extra", []int64{1}),
			entity.NewColumnFloatVector("vector", 128, generateFloatVector(1, 128)),
		)

		s.NoError(err)
		s.Equal(1, r.Len())
	})

	s.Run("missing_field_with_default_value", func() {
		defer s.resetMock()
		s.setupHasCollection(testCollectionName)
		s.setupHasPartition(testCollectionName, "partition_1")

		s.setupDescribeCollection(testCollectionName, entity.NewSchema().
			WithField(entity.NewField().WithIsPrimaryKey(true).WithIsAutoID(true).WithName("ID").WithDataType(entity.FieldTypeInt64)).
			WithField(entity.NewField().WithName("default_value").WithDataType(entity.FieldTypeInt64).WithDefaultValueLong(1)).
			WithField(entity.NewField().WithName("vector").WithDataType(entity.FieldTypeFloatVector).WithTypeParams(entity.TypeParamDim, "128")),
		)

		s.mock.EXPECT().Upsert(mock.Anything, mock.AnythingOfType("*milvuspb.UpsertRequest")).
			Run(func(ctx context.Context, req *milvuspb.UpsertRequest) {
				var data *schemapb.FieldData
				for _, d := range req.GetFieldsData() {
					if d.FieldName == "default_value" {
						data = d
					}
				}
				s.Equal([]bool{false}, data.ValidData)
				s.Equal(0, len(data.GetScalars().GetLongData().Data))
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

		r, err := c.Upsert(ctx, testCollectionName, "partition_1",
			entity.NewColumnFloatVector("vector", 128, generateFloatVector(1, 128)),
		)
		s.NoError(err)
		s.Equal(1, r.Len())
	})

	s.Run("missing_field_with_nullable", func() {
		defer s.resetMock()
		s.setupHasCollection(testCollectionName)
		s.setupHasPartition(testCollectionName, "partition_1")

		s.setupDescribeCollection(testCollectionName, entity.NewSchema().
			WithField(entity.NewField().WithIsPrimaryKey(true).WithIsAutoID(true).WithName("ID").WithDataType(entity.FieldTypeInt64)).
			WithField(entity.NewField().WithName("nullable_fid").WithDataType(entity.FieldTypeInt64).WithNullable(true)).
			WithField(entity.NewField().WithName("vector").WithDataType(entity.FieldTypeFloatVector).WithTypeParams(entity.TypeParamDim, "128")),
		)

		s.mock.EXPECT().Upsert(mock.Anything, mock.AnythingOfType("*milvuspb.UpsertRequest")).
			Run(func(ctx context.Context, req *milvuspb.UpsertRequest) {
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

		r, err := c.Upsert(ctx, testCollectionName, "partition_1",
			entity.NewColumnFloatVector("vector", 128, generateFloatVector(1, 128)),
		)
		s.NoError(err)
		s.Equal(1, r.Len())
	})

	s.Run("insert_with_nullable_column", func() {
		defer s.resetMock()
		s.setupHasCollection(testCollectionName)
		s.setupHasPartition(testCollectionName, "partition_1")

		s.setupDescribeCollection(testCollectionName, entity.NewSchema().
			WithField(entity.NewField().WithIsPrimaryKey(true).WithIsAutoID(true).WithName("ID").WithDataType(entity.FieldTypeInt64)).
			WithField(entity.NewField().WithName("nullable_fid").WithDataType(entity.FieldTypeInt64).WithNullable(true)).
			WithField(entity.NewField().WithName("vector").WithDataType(entity.FieldTypeFloatVector).WithTypeParams(entity.TypeParamDim, "128")),
		)

		s.mock.EXPECT().Upsert(mock.Anything, mock.AnythingOfType("*milvuspb.UpsertRequest")).
			Run(func(ctx context.Context, req *milvuspb.UpsertRequest) {
				var data *schemapb.FieldData
				for _, d := range req.GetFieldsData() {
					if d.FieldName == "nullable_fid" {
						data = d
					}
				}
				s.Equal([]bool{false, true}, data.ValidData)
				s.Equal([]int64{1}, data.GetScalars().GetLongData().Data)
				s.Equal(2, len(req.GetFieldsData()))
			}).Return(&milvuspb.MutationResult{
			Status: &commonpb.Status{},
			IDs: &schemapb.IDs{
				IdField: &schemapb.IDs_IntId{
					IntId: &schemapb.LongArray{
						Data: []int64{1, 2},
					},
				},
			},
		}, nil)
		value := make([]int64, 2)
		value[1] = 1
		validValue := make([]bool, 2)
		validValue[1] = true
		r, err := c.Upsert(ctx, testCollectionName, "partition_1",
			entity.NewNullableColumnInt64("nullable_fid", value, validValue),
			entity.NewColumnFloatVector("vector", 128, generateFloatVector(2, 128)),
		)
		s.NoError(err)
		s.Equal(2, r.Len())
	})

	s.Run("insert_with_nullable_column_with_default_value", func() {
		defer s.resetMock()
		s.setupHasCollection(testCollectionName)
		s.setupHasPartition(testCollectionName, "partition_1")

		s.setupDescribeCollection(testCollectionName, entity.NewSchema().
			WithField(entity.NewField().WithIsPrimaryKey(true).WithIsAutoID(true).WithName("ID").WithDataType(entity.FieldTypeInt64)).
			WithField(entity.NewField().WithName("nullable_fid").WithDataType(entity.FieldTypeInt64).WithNullable(true).WithDefaultValueLong(10)).
			WithField(entity.NewField().WithName("vector").WithDataType(entity.FieldTypeFloatVector).WithTypeParams(entity.TypeParamDim, "128")),
		)

		s.mock.EXPECT().Upsert(mock.Anything, mock.AnythingOfType("*milvuspb.UpsertRequest")).
			Run(func(ctx context.Context, req *milvuspb.UpsertRequest) {
				s.Equal(2, len(req.GetFieldsData()))
			}).Return(&milvuspb.MutationResult{
			Status: &commonpb.Status{},
			IDs: &schemapb.IDs{
				IdField: &schemapb.IDs_IntId{
					IntId: &schemapb.LongArray{
						Data: []int64{1, 2},
					},
				},
			},
		}, nil)
		value := make([]int64, 2)
		value[1] = 1
		validValue := make([]bool, 2)
		validValue[1] = true
		r, err := c.Upsert(ctx, testCollectionName, "partition_1",
			entity.NewNullableColumnInt64("nullable_fid", value, validValue),
			entity.NewColumnFloatVector("vector", 128, generateFloatVector(2, 128)),
		)
		s.NoError(err)
		s.Equal(2, r.Len())
	})
}

func TestWrite(t *testing.T) {
	suite.Run(t, new(InsertSuite))
	suite.Run(t, new(UpsertSuite))
}

type DeleteSuite struct {
	MockSuiteBase
}

func (s *DeleteSuite) TestDeleteByPks() {
	c := s.client
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	collectionName := fmt.Sprintf("coll_%s", randStr(6))
	partitionName := fmt.Sprintf("part_%s", randStr(6))

	s.Run("normal_case", func() {
		defer s.resetMock()

		s.setupDescribeCollection(collectionName, entity.NewSchema().
			WithField(entity.NewField().WithIsPrimaryKey(true).WithIsAutoID(true).WithName("ID").WithDataType(entity.FieldTypeInt64)).
			WithField(entity.NewField().WithName("vector").WithDataType(entity.FieldTypeFloatVector).WithTypeParams(entity.TypeParamDim, "128")),
		)
		s.mock.EXPECT().Delete(mock.Anything, mock.AnythingOfType("*milvuspb.DeleteRequest")).RunAndReturn(func(ctx context.Context, dr *milvuspb.DeleteRequest) (*milvuspb.MutationResult, error) {
			s.Equal(collectionName, dr.GetCollectionName())
			s.Equal(partitionName, dr.GetPartitionName())
			return &milvuspb.MutationResult{
				Status: s.getSuccessStatus(),
			}, nil
		}).Once()
		err := c.DeleteByPks(ctx, collectionName, partitionName, entity.NewColumnInt64("ID", []int64{1, 2, 3}))
		s.NoError(err)
	})

	s.Run("bad_requests", func() {
		defer s.resetMock()

		s.setupDescribeCollection(collectionName, entity.NewSchema().
			WithField(entity.NewField().WithIsPrimaryKey(true).WithIsAutoID(true).WithName("ID").WithDataType(entity.FieldTypeInt64)).
			WithField(entity.NewField().WithName("vector").WithDataType(entity.FieldTypeFloatVector).WithTypeParams(entity.TypeParamDim, "128")),
		)

		s.Run("zero_length_pks", func() {
			err := c.DeleteByPks(ctx, collectionName, "", entity.NewColumnInt64("ID", []int64{}))
			s.Error(err)
		})

		s.Run("pk_type_not_valid", func() {
			err := c.DeleteByPks(ctx, collectionName, "", entity.NewColumnBool("ID", []bool{true, false}))
			s.Error(err)
		})

		s.Run("pk_name_not_match", func() {
			err := c.DeleteByPks(ctx, collectionName, "", entity.NewColumnInt64("pk_", []int64{100, 200}))
			s.Error(err)
		})
	})

	s.Run("server_error", func() {
		defer s.resetMock()

		s.setupDescribeCollection(collectionName, entity.NewSchema().
			WithField(entity.NewField().WithIsPrimaryKey(true).WithIsAutoID(true).WithName("ID").WithDataType(entity.FieldTypeInt64)).
			WithField(entity.NewField().WithName("vector").WithDataType(entity.FieldTypeFloatVector).WithTypeParams(entity.TypeParamDim, "128")),
		)
		s.mock.EXPECT().Delete(mock.Anything, mock.AnythingOfType("")).RunAndReturn(func(ctx context.Context, dr *milvuspb.DeleteRequest) (*milvuspb.MutationResult, error) {
			s.Equal(collectionName, dr.GetCollectionName())
			s.Equal(partitionName, dr.GetPartitionName())
			return nil, errors.New("mocked")
		}).Once()
		err := c.DeleteByPks(ctx, collectionName, partitionName, entity.NewColumnInt64("ID", []int64{1, 2, 3}))
		s.Error(err)
	})
}

func (s *DeleteSuite) TestDelete() {
	c := s.client
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	collectionName := fmt.Sprintf("coll_%s", randStr(6))
	partitionName := fmt.Sprintf("part_%s", randStr(6))

	s.Run("normal_case", func() {
		defer s.resetMock()
		expr := fmt.Sprintf("tag in [%d, %d, %d]", rand.Int31n(10), rand.Int31n(20), rand.Int31n(30))

		s.mock.EXPECT().Delete(mock.Anything, mock.Anything).RunAndReturn(func(ctx context.Context, dr *milvuspb.DeleteRequest) (*milvuspb.MutationResult, error) {
			s.Equal(collectionName, dr.GetCollectionName())
			s.Equal(partitionName, dr.GetPartitionName())
			s.Equal(expr, dr.GetExpr())
			return &milvuspb.MutationResult{
				Status: s.getSuccessStatus(),
			}, nil
		}).Once()

		err := c.Delete(ctx, collectionName, partitionName, expr)
		s.NoError(err)
	})

	s.Run("server_error", func() {
		expr := "tag > 50"
		defer s.resetMock()
		s.mock.EXPECT().Delete(mock.Anything, mock.Anything).RunAndReturn(func(ctx context.Context, dr *milvuspb.DeleteRequest) (*milvuspb.MutationResult, error) {
			s.Equal(collectionName, dr.GetCollectionName())
			s.Equal(partitionName, dr.GetPartitionName())
			s.Equal(expr, dr.GetExpr())
			return &milvuspb.MutationResult{
				Status: s.getStatus(commonpb.ErrorCode_UnexpectedError, "mocked"),
			}, nil
		}).Once()

		err := c.Delete(ctx, collectionName, partitionName, expr)
		s.Error(err)
	})
}

func TestDelete(t *testing.T) {
	suite.Run(t, new(DeleteSuite))
}
